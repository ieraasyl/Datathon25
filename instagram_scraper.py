#!/usr/bin/env python3
"""
instagram_scraper.py

Extracts Instagram post comments from specific accounts.
Handles login, sessions, retries, and saves results into JSON files.

Workflow:
1. Resolves Instagram user IDs from usernames (altel_kz, tele2kazakhstan).
2. Fetches latest posts for each account.
3. Extracts comments with retries and error handling.
4. Saves two JSON files into data/raw/:
   - altel_instagram_TIMESTAMP.json
   - tele2_instagram_TIMESTAMP.json
"""

import os
import sys
import json
import random
import time
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

from instagrapi import Client

# Config
load_dotenv()
MAX_POSTS = 50 # 0 - all posts
SESSION_DIR = "./sessions"
OUTPUT_DIR = "./data/raw"


class SimpleInstagramScraper:
    def __init__(self):
        self.client = Client()
        os.makedirs(SESSION_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def login(self, username: str, password: str) -> bool:
        """Login and save session."""
        try:
            print(f"🔑 Logging in as {username}...")
            self.client.login(username, password)
            self.client.dump_settings(f"{SESSION_DIR}/{username}.json")
            print("✅ Login successful")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

    def setup_session(self, username: str) -> bool:
        """Try to load an existing session."""
        session_file = f"{SESSION_DIR}/{username}.json"
        if os.path.exists(session_file):
            try:
                self.client.load_settings(session_file)
                print(f"✅ Loaded session from {session_file}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load session: {e}")
        return False

    def get_user_post_codes(self, user_id: str, max_posts: int) -> List[str]:
        """Fetch recent post codes from user feed using private_request (bypasses instagrapi validation)."""
        medias = []
        next_max_id = None
        fetched = 0
        page = 0

        print(f"\n🎯 Fetching posts for user {user_id}...")

        while True:
            page += 1
            params = {"count": 50}
            if next_max_id:
                params["max_id"] = next_max_id

            try:
                resp = self.client.private_request(f"feed/user/{user_id}/", params=params)
            except Exception as e:
                print(f"❌ Error fetching user feed for {user_id}: {e}")
                break

            items = resp.get("items", [])
            if not items:
                print(f"⚠️ No more items returned at page {page}")
                break

            for media in items:
                try:
                    medias.append(media["code"])
                    fetched += 1
                    if max_posts > 0 and fetched >= max_posts:
                        print(f"✅ Collected {len(medias)} posts for user {user_id}")
                        return medias
                except KeyError:
                    continue

            print(f"📄 Page {page}: {len(items)} posts fetched (total {fetched})")

            next_max_id = resp.get("next_max_id")
            if not next_max_id:
                break

        print(f"✅ Collected {len(medias)} posts for user {user_id}")
        return medias

    def extract_comments(
        self, post: str, max_comments: int = 0, retries: int = 3
    ) -> Dict[str, Any]:
        """
        Extract comments from an Instagram post.
        Handles photos and reels gracefully.
        """
        result: Dict[str, Any] = {
            "post_url": None,
            "comments": [],
            "total_comments": 0,
            "extracted_at": datetime.now().isoformat(),
        }

        # Normalize input
        if post.startswith("http"):
            url = post
        else:
            url = f"https://www.instagram.com/p/{post}/"
        result["post_url"] = url

        try:
            try:
                media_pk = self.client.media_pk_from_url(url)
                media_id = self.client.media_id(media_pk)
            except Exception as pk_error:
                # Handle reels with invalid metadata
                if "clips_metadata" in str(pk_error).lower():
                    result["note"] = "Reel/Video with invalid clips_metadata (skipped)"
                    print(f"ℹ️ Skipping Reel with invalid metadata: {url}")
                    return result
                raise

            # Retry fetching comments
            raw_comments = []
            last_error = None
            for attempt in range(1, retries + 1):
                try:
                    raw_comments = self.client.media_comments(
                        media_id, amount=(max_comments or 0)
                    )
                    break
                except Exception as e:
                    last_error = e
                    wait = random.uniform(2, 5) * attempt
                    print(
                        f"⚠️ Error fetching comments (try {attempt}/{retries}): {e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
            else:
                result["error"] = f"Failed after {retries} retries: {last_error}"
                return result

            # Parse comments
            comments = []
            for c in raw_comments:
                try:
                    created_at = getattr(c, "created_at_utc", None)
                    comments.append(
                        {
                            "text": getattr(c, "text", ""),
                            "username": getattr(c.user, "username", None),
                            "like_count": getattr(c, "like_count", 0),
                            "created_at_utc": created_at.isoformat()
                            if created_at
                            else None,
                        }
                    )
                except Exception:
                    continue

            result["comments"] = comments
            result["total_comments"] = len(comments)

            print(f"✅ Extracted {len(comments)} comments from {url}")
            return result

        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            print(f"❌ Error extracting comments from {url}: {e}")
            return result

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of results."""
        total_posts = len(results)
        total_comments = sum(r.get("total_comments", 0) for r in results)
        failed_posts = sum(1 for r in results if "error" in r)
        print("\n📊 Summary")
        print("=" * 40)
        print(f"Total posts processed: {total_posts}")
        print(f"Total comments extracted: {total_comments}")
        print(f"Failed posts: {failed_posts}")


def main():
    print("🚀 Instagram Comments Extractor")
    print("=" * 50)

    # Get credentials
    username = os.getenv("INSTAGRAM_USERNAME")
    if not username:
        sys.exit("❌ USERNAME environment variable is not set")
    scraper = SimpleInstagramScraper()

    if not scraper.setup_session(username):
        password = input("🔐 Instagram password: ").strip()
        if not scraper.login(username, password):
            sys.exit("❌ Could not login")
    else:
        try:
            scraper.client.get_timeline_feed()
            print("✅ Session is valid")
        except:
            password = input("🔐 Session expired. Instagram password: ").strip()
            if not scraper.login(username, password):
                sys.exit("❌ Could not login")

    # Resolve both user IDs
    print("\n🔍 Resolving Instagram user IDs...")
    #altel_user_id = str(1032022674)
    tele2_user_id = str(451489228)
    # altel_user_id = scraper.client.user_id_from_username("altel_kz")
    # tele2_user_id = scraper.client.user_id_from_username("tele2kazakhstan")
    #print(f"✅ altel_kz → {altel_user_id}")
    print(f"✅ tele2kazakhstan → {tele2_user_id}")

    def process_account(user_id: str, label: str):
        print(f"\n🎯 Fetching posts for {label} ({user_id})")
        post_codes = scraper.get_user_post_codes(user_id, MAX_POSTS)
        if not post_codes:
            print(f"❌ No posts found for {label}")
            return

        print(f"\n🚀 Extracting comments from {len(post_codes)} posts ({label})...")
        results = []
        for i, code in enumerate(post_codes, 1):
            if i == 22 or i == 33:  # Known problematic posts
                print(f"\nSkipping post {i}, 8k and 800 comments (to avoid long processing)")
                continue
            print(f"\n🎯 Processing {label} post {i}/{len(post_codes)}: {code}")
            data = scraper.extract_comments(code)
            results.append(data)

        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"{label}_instagram_{timestamp}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved results to {filename}")
            scraper.print_summary(results)
        else:
            print(f"❌ No results for {label}")

    # Run for both accounts
    #process_account(altel_user_id, "altel")
    process_account(tele2_user_id, "tele2")


if __name__ == "__main__":
    main()
