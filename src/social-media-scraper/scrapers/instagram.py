import re
import requests
from typing import List, Optional
from fastapi import HTTPException
from .base import BaseScraper
from ..models.comments import EnhancedComment
import logging

logger = logging.getLogger(__name__)

class InstagramScraper(BaseScraper):
    """Instagram comment scraper using Instagram Basic Display API"""
    
    def __init__(self, access_token: str):
        super().__init__(access_token)
        if not access_token:
            raise ValueError("Instagram access token is required")
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def extract_post_id(self, url: str) -> Optional[str]:
        """Extract Instagram shortcode from URL"""
        patterns = [
            r"/p/([A-Za-z0-9_-]+)/",
            r"/reel/([A-Za-z0-9_-]+)/",
            r"/tv/([A-Za-z0-9_-]+)/"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _resolve_shortcode_to_media_id(self, shortcode: str) -> str:
        """Convert Instagram shortcode to media ID"""
        try:
            response = requests.get(
                f"{self.base_url}/ig_shortcode({shortcode})",
                params={"access_token": self.api_key},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                raise HTTPException(status_code=400, detail=f"Instagram API error: {data['error']['message']}")
            
            return data.get('id')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error resolving Instagram shortcode: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Instagram shortcode resolution error: {str(e)}")
    
    async def scrape_comments(self, url: str, max_comments: int, business_name: str) -> List[EnhancedComment]:
        """Scrape Instagram comments"""
        shortcode = self.extract_post_id(url)
        if not shortcode:
            raise HTTPException(status_code=400, detail="Invalid Instagram URL")
        
        # Resolve shortcode to media ID
        media_id = self._resolve_shortcode_to_media_id(shortcode)
        if not media_id:
            raise HTTPException(status_code=400, detail="Could not resolve Instagram media ID")
        
        comments = []
        api_url = f"{self.base_url}/{media_id}/comments"
        params = {
            "access_token": self.api_key,
            "limit": min(100, max_comments),
            "fields": "id,from,text,timestamp,like_count,replies"
        }
        
        fetched = 0
        
        try:
            while fetched < max_comments:
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    raise HTTPException(status_code=400, detail=f"Instagram API error: {data['error']['message']}")
                
                for item in data.get('data', []):
                    comment_text = item.get('text', '')
                    from_data = item.get('from', {})
                    
                    comment = EnhancedComment(
                        id=item.get('id'),
                        author=from_data.get('username', 'Unknown'),
                        author_id=from_data.get('id'),
                        timestamp=item.get('timestamp'),
                        text=comment_text,
                        platform='instagram',
                        post_id=media_id,
                        post_url=url,
                        business_name=business_name,
                        likes_count=item.get('like_count', 0),
                        replies_count=len(item.get('replies', {}).get('data', []))
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
                params = {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error scraping Instagram comments: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Instagram scraping error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error scraping Instagram: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected Instagram error: {str(e)}")
        
        return comments
