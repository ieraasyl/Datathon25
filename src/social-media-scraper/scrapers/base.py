from abc import ABC, abstractmethod
from typing import List
import logging
from ..models.comments import EnhancedComment

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Base class for all social media scrapers"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.logger = logger
    
    @abstractmethod
    async def scrape_comments(self, url: str, max_comments: int, business_name: str) -> List[EnhancedComment]:
        """Abstract method for scraping comments"""
        pass
    
    @abstractmethod
    def extract_post_id(self, url: str) -> str:
        """Abstract method for extracting post ID from URL"""
        pass
    
    def validate_url(self, url: str) -> bool:
        """Validate if URL is supported by this scraper"""
        try:
            return bool(self.extract_post_id(url))
        except:
            return False