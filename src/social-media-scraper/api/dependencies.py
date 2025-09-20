from fastapi import Depends, HTTPException, status
from typing import Optional
import os
import logging
from ..config.settings import settings
from ..scrapers.youtube import YouTubeScraper
from ..scrapers.facebook import FacebookScraper
from ..scrapers.instagram import InstagramScraper
from ..scrapers.vk import VKScraper

logger = logging.getLogger(__name__)

class ScraperFactory:
    """Factory for creating scraper instances"""
    
    def __init__(self):
        self._scrapers = {}
        self._initialize_scrapers()
    
    def _initialize_scrapers(self):
        """Initialize available scrapers based on configuration"""
        try:
            if settings.youtube_api_key:
                self._scrapers['youtube'] = YouTubeScraper(settings.youtube_api_key)
                logger.info("YouTube scraper initialized")
        except Exception as e:
            logger.warning(f"YouTube scraper initialization failed: {str(e)}")
        
        try:
            if settings.facebook_token:
                self._scrapers['facebook'] = FacebookScraper(settings.facebook_token)
                logger.info("Facebook scraper initialized")
        except Exception as e:
            logger.warning(f"Facebook scraper initialization failed: {str(e)}")
        
        try:
            if settings.instagram_token:
                self._scrapers['instagram'] = InstagramScraper(settings.instagram_token)
                logger.info("Instagram scraper initialized")
        except Exception as e:
            logger.warning(f"Instagram scraper initialization failed: {str(e)}")
        
        try:
            # VK scraper can work without token for public posts
            self._scrapers['vk'] = VKScraper(settings.vk_token)
            logger.info("VK scraper initialized")
        except Exception as e:
            logger.warning(f"VK scraper initialization failed: {str(e)}")
    
    def get_scraper(self, platform: str):
        """Get scraper for specific platform"""
        scraper = self._scrapers.get(platform)
        if not scraper:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Scraper for platform '{platform}' is not available"
            )
        return scraper
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platforms"""
        return list(self._scrapers.keys())
    
    def is_platform_available(self, platform: str) -> bool:
        """Check if platform is available"""
        return platform in self._scrapers

# Global scraper factory instance
scraper_factory = ScraperFactory()

def get_scraper_factory() -> ScraperFactory:
    """Dependency to get scraper factory"""
    return scraper_factory

def validate_platform_availability(platforms: Optional[List[str]] = None):
    """Validate that requested platforms are available"""
    if not platforms:
        return
    
    available_platforms = scraper_factory.get_available_platforms()
    unavailable = [p for p in platforms if p not in available_platforms]
    
    if unavailable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Platforms not available: {', '.join(unavailable)}. Available: {', '.join(available_platforms)}"
        )