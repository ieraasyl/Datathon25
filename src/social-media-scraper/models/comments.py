from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class EnhancedComment:
    """Enhanced comment structure with rich metadata"""
    id: str
    author: str
    author_id: Optional[str]
    timestamp: str
    text: str
    platform: str
    post_id: str
    post_url: str
    business_name: str
    likes_count: Optional[int] = 0
    replies_count: Optional[int] = 0
    is_verified: Optional[bool] = False
    language: Optional[str] = None
    sentiment: Optional[str] = None
    word_count: Optional[int] = 0
    has_mentions: Optional[bool] = False
    has_hashtags: Optional[bool] = False
    scraped_at: Optional[str] = None

    def __post_init__(self):
        """Post-initialization processing"""
        if self.text:
            self.word_count = len(self.text.split())
            self.has_mentions = '@' in self.text
            self.has_hashtags = '#' in self.text
        
        if not self.scraped_at:
            self.scraped_at = datetime.utcnow().isoformat() + "Z"
