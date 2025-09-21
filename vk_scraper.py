#!/usr/bin/env python3
"""
VKontakte Comments Extractor

This script extracts all comments from the altel5g VK community posts
and saves them in JSON format following the provided schema.

Requirements:
- pip install requests
- VK API access token (can be obtained from https://vkhost.github.io/)
"""

import requests
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VKCommentsExtractor:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.api_version = "5.131"
        self.base_url = "https://api.vk.com/method/"

    def _make_request(self, method: str, params: Dict[str, Any]) -> Any:
        params.update({
            'access_token': self.access_token,
            'v': self.api_version
        })
        url = f"{self.base_url}{method}"
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                raise Exception(f"VK API Error: {data['error']}")
            return data.get('response', {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"API error: {e}")
            raise

    def get_community_posts(self, domain: str, count: int = 100, offset: int = 0) -> Dict[str, Any]:
        params = {
            'domain': domain if not domain.startswith('-') else None,
            'owner_id': domain if domain.startswith('-') else None,
            'count': min(count, 100),
            'offset': offset,
            'extended': 1
        }
        params = {k: v for k, v in params.items() if v is not None}
        logger.info(f"Fetching posts from {domain}, offset: {offset}")
        return self._make_request('wall.get', params)

    def get_post_comments(self, owner_id: str, post_id: int, offset: int = 0, count: int = 100) -> Dict[str, Any]:
        params = {
            'owner_id': owner_id,
            'post_id': post_id,
            'offset': offset,
            'count': min(count, 100),
            'sort': 'asc',
            'extended': 1,
            'fields': 'first_name,last_name'
        }
        return self._make_request('wall.getComments', params)

    def extract_all_comments(self, domain: str, max_posts: int = 0) -> List[Dict[str, Any]]:
        extracted_data = []
        posts_processed = 0
        offset = 0
        batch_size = 100

        logger.info(f"Starting extraction from {domain}")

        while True:
            try:
                posts_data = self.get_community_posts(domain, batch_size, offset)
                posts = posts_data.get('items', [])
                if not posts:
                    logger.info("No more posts to process")
                    break

                for post in posts:
                    if max_posts and posts_processed >= max_posts:
                        logger.info(f"Reached maximum posts limit: {max_posts}")
                        return extracted_data

                    post_id = post['id']
                    owner_id = post['owner_id']
                    post_url = f"https://vk.com/wall{owner_id}_{post_id}"

                    logger.info(f"🚀 Processing post {posts_processed + 1}: {post_id}")

                    all_comments = []
                    comments_offset = 0
                    total_comments = 0

                    while True:
                        try:
                            comments_data = self.get_post_comments(owner_id, post_id, comments_offset, 100)
                            comments = comments_data.get('items', [])
                            profiles = {p['id']: p for p in comments_data.get('profiles', [])}
                            groups = {g['id']: g for g in comments_data.get('groups', [])}
                            total_comments = comments_data.get('count', 0)

                            if not comments:
                                break

                            for comment in comments:
                                from_id = comment['from_id']
                                if from_id > 0:  # User
                                    user = profiles.get(from_id, {})
                                    username = f"{user.get('first_name', 'Unknown')} {user.get('last_name', 'User')}"
                                else:  # Group
                                    group = groups.get(abs(from_id), {})
                                    username = group.get('name', 'Unknown Group')

                                comment_data = {
                                    "text": comment.get('text', ''),
                                    "username": username,
                                    "like_count": comment.get('likes', {}).get('count', 0),
                                    "created_at_utc": datetime.fromtimestamp(
                                        comment['date'], timezone.utc
                                    ).isoformat().replace('+00:00', 'Z')
                                }
                                all_comments.append(comment_data)

                            comments_offset += len(comments)
                            time.sleep(0.34)

                        except Exception as e:
                            logger.error(f"Error getting comments for post {post_id}: {e}")
                            break

                    post_entry = {
                        "post_url": post_url,
                        "comments": all_comments,
                        "total_comments": total_comments,
                        "extracted_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                    }

                    extracted_data.append(post_entry)
                    posts_processed += 1
                    logger.info(f"✅ Extracted {len(all_comments)} comments from post {post_id}")

                offset += len(posts)
                time.sleep(0.34)

            except Exception as e:
                logger.error(f"Error processing posts batch at offset {offset}: {e}")
                break

        logger.info(f"Extraction complete. Processed {posts_processed} posts")
        return extracted_data

    def save_to_json(self, data: List[Dict[str, Any]], filename: str) -> str:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")
        return filename


def main():
    ACCESS_TOKEN = "vk1.a.JB2tw95SCNP6znmnjY4nOZxKaUDHqX0NtONvha2pzEEg8pdJvNG1IuQcJCDZOma0ZxAkkP4wLqXlvidODlhq_cLloiEOMpYmWj8-2YYhQE5KjGGOoB83tvy8glyyY6ZevKlHsGMTPuKVyo2Xdg6GaGsjouVDV0cVue3VQZ1Ce0NHjxBvsShgXcYhu-WP52cqfAgRMc06B24ZQmCHbhUVSg"  # replace with your VK token
    MAX_POSTS = 50  # 0 = all posts

    extractor = VKCommentsExtractor(ACCESS_TOKEN)

    # --- altel5g extraction ---
    ALTEL_DOMAIN = "altel5g"
    altel_filename = f"data/raw/altel_vk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"\n🎯 Extracting comments from: {ALTEL_DOMAIN}")
    altel_data = extractor.extract_all_comments(ALTEL_DOMAIN, MAX_POSTS)
    extractor.save_to_json(altel_data, altel_filename)

    # --- tele2kz extraction (disabled due to comments being off) ---
    """
    TELE2_DOMAIN = "tele2kz"
    tele2_filename = f"data/raw/tele2_vk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print(f"\n🎯 Extracting comments from: {TELE2_DOMAIN} (disabled)")
    tele2_data = extractor.extract_all_comments(TELE2_DOMAIN, MAX_POSTS)
    extractor.save_to_json(tele2_data, tele2_filename)
    """

    # Summary
    total_posts = len(altel_data)
    total_comments = sum(post['total_comments'] for post in altel_data)
    print(f"\n{'='*40}")
    print(f"Extraction Summary:")
    print(f"Total posts processed: {total_posts}")
    print(f"Total comments extracted: {total_comments}")
    print(f"Output file: {altel_filename}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()