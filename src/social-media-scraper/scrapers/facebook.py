import re
import requests
from typing import List, Optional
from fastapi import HTTPException
from .base import BaseScraper
from ..models.comments import EnhancedComment
import logging

logger = logging.getLogger(__name__)

class FacebookScraper(BaseScraper):
    """Facebook comment scraper using Graph API"""
    
    def __init__(self, access_token: str):
        super().__init__(access_token)
        if not access_token:
            raise ValueError("Facebook access token is required")
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def extract_post_id(self, url: str) -> Optional[str]:
        """Extract Facebook post ID from URL"""
        patterns = [
            r"posts/(\d+)",
            r"permalink\.php\?story_fbid=(\d+)",
            r"photo\.php\?fbid=(\d+)",
            r"/(\d+)/posts/(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1) if len(match.groups()) == 1 else match.group(2)
        return None
    
    async def scrape_comments(self, url: str, max_comments: int, business_name: str) -> List[EnhancedComment]:
        """Scrape Facebook comments"""
        post_id = self.extract_post_id(url)
        if not post_id:
            raise HTTPException(status_code=400, detail="Invalid Facebook URL")
        
        comments = []
        api_url = f"{self.base_url}/{post_id}/comments"
        params = {
            "access_token": self.api_key,
            "limit": min(100, max_comments),
            "fields": "id,from,message,created_time,like_count,comment_count,attachment"
        }
        
        fetched = 0
        
        try:
            while fetched < max_comments:
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    raise HTTPException(status_code=400, detail=f"Facebook API error: {data['error']['message']}")
                
                for item in data.get('data', []):
                    comment_text = item.get('message', '')
                    from_data = item.get('from', {})
                    
                    comment = EnhancedComment(
                        id=item.get('id'),
                        author=from_data.get('name', 'Unknown'),
                        author_id=from_data.get('id'),
                        timestamp=item.get('created_time'),
                        text=comment_text,
                        platform='facebook',
                        post_id=post_id,
                        post_url=url,
                        business_name=business_name,
                        likes_count=item.get('like_count', 0),
                        replies_count=item.get('comment_count', 0)
                    )
                    
                    comments.append(comment)
                    fetched += 1
                    
                    if fetched >= max_comments:
                        break
                
                # Check for next page
                paging = data.get('paging', {})
                if fetched >= max_comments or 'next' not in paging:
                    break
                
                api_url = paging['next']
                params = {}  # Next URL already contains all parameters
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Facebook comments: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Facebook scraping error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error scraping Facebook: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected Facebook error: {str(e)}")
        
        return comments
