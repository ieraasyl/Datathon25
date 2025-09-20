import re
from typing import List, Optional
from googleapiclient.discovery import build
from fastapi import HTTPException
from .base import BaseScraper
from ..models.comments import EnhancedComment

class YouTubeScraper(BaseScraper):
    """YouTube comment scraper"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if not api_key:
            raise ValueError("YouTube API key is required")
        self.service = build("youtube", "v3", developerKey=api_key)
    
    def extract_post_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r"v=([A-Za-z0-9_-]{11})",
            r"youtu\.be/([A-Za-z0-9_-]{11})",
            r"youtube.com/embed/([A-Za-z0-9_-]{11})",
            r"youtube.com/watch\?v=([A-Za-z0-9_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def scrape_comments(self, url: str, max_comments: int, business_name: str) -> List[EnhancedComment]:
        """Scrape YouTube comments"""
        video_id = self.extract_post_id(url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
        
        comments = []
        next_page_token = None
        fetched = 0
        
        try:
            while fetched < max_comments:
                response = self.service.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_comments - fetched),
                    pageToken=next_page_token,
                    textFormat="plainText"
                ).execute()
                
                for item in response.get('items', []):
                    snippet = item['snippet']['topLevelComment']['snippet']
                    
                    comment = EnhancedComment(
                        id=item['id'],
                        author=snippet.get('authorDisplayName', 'Unknown'),
                        author_id=snippet.get('authorChannelId', {}).get('value'),
                        timestamp=snippet.get('publishedAt'),
                        text=snippet.get('textDisplay', ''),
                        platform='youtube',
                        post_id=video_id,
                        post_url=url,
                        business_name=business_name,
                        likes_count=snippet.get('likeCount', 0),
                        replies_count=item.get('replies', {}).get('totalReplyCount', 0)
                    )
                    
                    comments.append(comment)
                    fetched += 1
                    
                    if fetched >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error scraping YouTube comments: {str(e)}")
            raise HTTPException(status_code=500, detail=f"YouTube scraping error: {str(e)}")
        
        return comments


