import re
from typing import List, Optional
from urllib.parse import urlparse
import validators as url_validators

class URLValidator:
    """URL validation utilities"""
    
    PLATFORM_PATTERNS = {
        'youtube': [
            r'(?:youtube\.com|youtu\.be)',
            r'youtube\.com/watch\?v=',
            r'youtu\.be/',
            r'youtube\.com/embed/'
        ],
        'facebook': [
            r'facebook\.com',
            r'fb\.com',
            r'/posts/',
            r'/photos/',
            r'permalink\.php'
        ],
        'instagram': [
            r'instagram\.com',
            r'/p/',
            r'/reel/',
            r'/tv/'
        ],
        'vk': [
            r'vk\.com',
            r'/wall',
            r'wall-?\d+_\d+'
        ]
    }
    
    @classmethod
    def is_valid_url(cls, url: str) -> bool:
        """Check if URL is valid"""
        try:
            return url_validators.url(url)
        except:
            return False
    
    @classmethod
    def detect_platform(cls, url: str) -> Optional[str]:
        """Detect platform from URL"""
        if not cls.is_valid_url(url):
            return None
        
        url_lower = url.lower()
        
        for platform, patterns in cls.PLATFORM_PATTERNS.items():
            if any(re.search(pattern, url_lower) for pattern in patterns):
                return platform
        
        return None
    
    @classmethod
    def validate_platform_url(cls, url: str, expected_platform: str) -> bool:
        """Validate URL belongs to expected platform"""
        detected_platform = cls.detect_platform(url)
        return detected_platform == expected_platform
    
    @classmethod
    def validate_batch_urls(cls, urls: List[str]) -> Dict[str, List[str]]:
        """Validate batch of URLs and categorize by platform"""
        result = {
            'valid': {},
            'invalid': [],
            'unsupported': []
        }
        
        for url in urls:
            if not cls.is_valid_url(url):
                result['invalid'].append(url)
                continue
            
            platform = cls.detect_platform(url)
            if not platform:
                result['unsupported'].append(url)
                continue
            
            if platform not in result['valid']:
                result['valid'][platform] = []
            result['valid'][platform].append(url)
        
        return result

class CommentValidator:
    """Comment data validation utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize comment text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    @staticmethod
    def is_valid_comment(comment_text: str, min_length: int = 1) -> bool:
        """Validate comment meets minimum requirements"""
        if not comment_text:
            return False
        
        cleaned_text = CommentValidator.clean_text(comment_text)
        return len(cleaned_text) >= min_length
    
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """Simple language detection (can be enhanced with external libraries)"""
        if not text:
            return None
        
        # Simple pattern matching for common languages
        if re.search(r'[а-яё]', text.lower()):
            return 'ru'
        elif re.search(r'[a-z]', text.lower()):
            return 'en'
        elif re.search(r'[äöüß]', text.lower()):
            return 'de'
        elif re.search(r'[àáâäèéêëìíîïòóôöùúûü]', text.lower()):
            return 'fr'
        elif re.search(r'[áéíóúñ]', text.lower()):
            return 'es'
        
        return 'unknown'
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """Extract @mentions from text"""
        if not text:
            return []
        
        mentions = re.findall(r'@(\w+)', text)
        return list(set(mentions))  # Remove duplicates
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """Extract #hashtags from text"""
        if not text:
            return []
        
        hashtags = re.findall(r'#(\w+)', text)
        return list(set(hashtags))  # Remove duplicates
