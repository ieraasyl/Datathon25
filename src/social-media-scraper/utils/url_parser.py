import re
from typing import Optional

class URLParser:
    """Utility class for parsing social media URLs"""
    
    @staticmethod
    def detect_platform(url: str) -> Optional[str]:
        """Auto-detect platform from URL"""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'vk.com' in url:
            return 'vk'
        elif 'facebook.com' in url:
            return 'facebook'
        elif 'instagram.com' in url:
            return 'instagram'
        return None
    
    @staticmethod
    def extract_youtube_id(url: str) -> Optional[str]:
        """Extract YouTube video ID"""
        patterns = [
            r"v=([A-Za-z0-9_-]{11})",
            r"youtu\.be/([A-Za-z0-9_-]{11})",
            r"youtube.com/embed/([A-Za-z0-9_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def extract_vk_post(url: str) -> Optional[dict]:
        """Extract VK post information"""
        patterns = [r"wall(-?\d+_\d+)", r"w=wall(-?\d+_\d+)"]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                parts = match.group(1).split('_')
                return {"owner_id": int(parts[0]), "post_id": int(parts[1])}
        return None

# =====================================
# src/social_scraper/config/settings.py
# =====================================


# =====================================
# src/social_scraper/main.py
# =====================================
