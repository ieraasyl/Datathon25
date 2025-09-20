import re
import datetime
from typing import List, Optional, Dict
from fastapi import HTTPException
from .base import BaseScraper
from ..models.comments import EnhancedComment
import logging

logger = logging.getLogger(__name__)

try:
    import vk_api
    VK_AVAILABLE = True
except ImportError:
    vk_api = None
    VK_AVAILABLE = False

class VKScraper(BaseScraper):
    """VK comment scraper"""
    
    def __init__(self, token: str = None):
        super().__init__(token)
        if not VK_AVAILABLE:
            raise ImportError("vk-api library is required for VK scraping")
        
        try:
            if token:
                self.session = vk_api.VkApi(token=token)
            else:
                # For public posts, we can try without token
                self.session = vk_api.VkApi()
            
            self.vk = self.session.get_api()
        except Exception as e:
            logger.error(f"VK API initialization error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"VK API initialization failed: {str(e)}")
    
    def extract_post_id(self, url: str) -> Optional[Dict[str, int]]:
        """Extract VK post information from URL"""
        patterns = [
            r"wall(-?\d+_\d+)",
            r"w=wall(-?\d+_\d+)",
            r"vk\.com/wall(-?\d+_\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                wall_id = match.group(1)
                parts = wall_id.split('_')
                return {
                    "owner_id": int(parts[0]),
                    "post_id": int(parts[1])
                }
        return None
    
    def _get_user_info(self, user_ids: List[int]) -> Dict[int, str]:
        """Get user information for author names"""
        try:
            if not user_ids:
                return {}
            
            # Filter out negative IDs (groups) and convert to positive
            positive_ids = [abs(uid) for uid in user_ids if uid > 0]
            negative_ids = [abs(uid) for uid in user_ids if uid < 0]
            
            result = {}
            
            # Get user info
            if positive_ids:
                users = self.vk.users.get(user_ids=positive_ids, fields='screen_name')
                for user in users:
                    user_id = user['id']
                    name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
                    result[user_id] = name or f"User {user_id}"
            
            # Get group info
            if negative_ids:
                groups = self.vk.groups.getById(group_ids=negative_ids, fields='screen_name')
                for group in groups:
                    group_id = -group['id']  # VK groups have negative IDs
                    result[group_id] = group.get('name', f"Group {group['id']}")
            
            return result
        except Exception as e:
            logger.warning(f"Could not fetch user info: {str(e)}")
            return {}
    
    async def scrape_comments(self, url: str, max_comments: int, business_name: str) -> List[EnhancedComment]:
        """Scrape VK comments"""
        post_info = self.extract_post_id(url)
        if not post_info:
            raise HTTPException(status_code=400, detail="Invalid VK URL")
        
        owner_id = post_info['owner_id']
        post_id = post_info['post_id']
        
        comments = []
        offset = 0
        fetched = 0
        
        # Collect all author IDs for batch user info retrieval
        author_ids = set()
        raw_comments = []
        
        try:
            # First pass: collect all comments and author IDs
            while fetched < max_comments:
                response = self.vk.wall.getComments(
                    owner_id=owner_id,
                    post_id=post_id,
                    count=min(100, max_comments - fetched),
                    offset=offset,
                    need_likes=1,
                    extended=0,
                    sort='asc'
                )
                
                items = response.get('items', [])
                if not items:
                    break
                
                for item in items:
                    raw_comments.append(item)
                    author_id = item.get('from_id')
                    if author_id:
                        author_ids.add(author_id)
                
                fetched += len(items)
                offset += len(items)
                
                if len(items) < 100:  # No more comments
                    break
            
            # Get author information
            user_info = self._get_user_info(list(author_ids))
            
            # Second pass: create EnhancedComment objects
            for item in raw_comments[:max_comments]:
                comment_text = item.get('text', '')
                author_id = item.get('from_id')
                author_name = user_info.get(author_id, f"User {author_id}" if author_id else "Unknown")
                
                # Convert Unix timestamp to ISO format
                timestamp = item.get('date')
                iso_timestamp = datetime.datetime.utcfromtimestamp(timestamp).isoformat() + 'Z' if timestamp else None
                
                comment = EnhancedComment(
                    id=str(item.get('id')),
                    author=author_name,
                    author_id=str(author_id) if author_id else None,
                    timestamp=iso_timestamp,
                    text=comment_text,
                    platform='vk',
                    post_id=f"{owner_id}_{post_id}",
                    post_url=url,
                    business_name=business_name,
                    likes_count=item.get('likes', {}).get('count', 0),
                    replies_count=item.get('thread', {}).get('count', 0)
                )
                
                comments.append(comment)
                
        except Exception as e:
            logger.error(f"Error scraping VK comments: {str(e)}")
            raise HTTPException(status_code=500, detail=f"VK scraping error: {str(e)}")
        
        return comments