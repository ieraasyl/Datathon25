import pytest
from src.social_scraper.utils.validators import URLValidator, CommentValidator

class TestURLValidator:
    
    def test_is_valid_url(self):
        """Test URL validation"""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.facebook.com/posts/123456789",
            "https://www.instagram.com/p/ABC123/",
            "https://vk.com/wall-12345_67890"
        ]
        
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "javascript:alert('xss')",
            ""
        ]
        
        for url in valid_urls:
            assert URLValidator.is_valid_url(url), f"Should be valid: {url}"
        
        for url in invalid_urls:
            assert not URLValidator.is_valid_url(url), f"Should be invalid: {url}"
    
    def test_detect_platform(self):
        """Test platform detection from URLs"""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
            ("https://www.facebook.com/posts/123456789", "facebook"),
            ("https://www.instagram.com/p/ABC123/", "instagram"),
            ("https://vk.com/wall-12345_67890", "vk"),
            ("https://twitter.com/user/status/123", None),  # Unsupported
            ("invalid-url", None)
        ]
        
        for url, expected_platform in test_cases:
            assert URLValidator.detect_platform(url) == expected_platform
    
    def test_validate_batch_urls(self):
        """Test batch URL validation"""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.facebook.com/posts/123456789",
            "invalid-url",
            "https://twitter.com/status/123",
            "https://vk.com/wall-12345_67890"
        ]
        
        result = URLValidator.validate_batch_urls(urls)
        
        assert "youtube" in result["valid"]
        assert "facebook" in result["valid"]
        assert "vk" in result["valid"]
        assert len(result["invalid"]) == 1
        assert len(result["unsupported"]) == 1

class TestCommentValidator:
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        test_cases = [
            ("  Hello   world  ", "Hello world"),
            ("Text\x00with\x08control\x1fchars", "Textwithcontrolchars"),
            ("\n\nMultiple\n\nlines\n\n", "Multiple lines"),
            ("", "")
        ]
        
        for input_text, expected in test_cases:
            assert CommentValidator.clean_text(input_text) == expected
    
    def test_detect_language(self):
        """Test basic language detection"""
        test_cases = [
            ("Hello world", "en"),
            ("Привет мир", "ru"),
            ("Hallo Welt", "de"),
            ("Bonjour monde", "fr"),
            ("Hola mundo", "es"),
            ("123456", "unknown"),
            ("", None)
        ]
        
        for text, expected_lang in test_cases:
            assert CommentValidator.detect_language(text) == expected_lang
    
    def test_extract_mentions(self):
        """Test mention extraction"""
        test_cases = [
            ("Hello @user1 and @user2", ["user1", "user2"]),
            ("No mentions here", []),
            ("@user @user", ["user"]),  # Should deduplicate
            ("Email: user@domain.com @mention", ["mention"])  # Should not match email
        ]
        
        for text, expected_mentions in test_cases:
            mentions = CommentValidator.extract_mentions(text)
            assert set(mentions) == set(expected_mentions)
    
    def test_extract_hashtags(self):
        """Test hashtag extraction"""
        test_cases = [
            ("Love this #awesome #video", ["awesome", "video"]),
            ("No hashtags here", []),
            ("#tag #tag", ["tag"]),  # Should deduplicate
            ("Price: $100 #sale", ["sale"])  # Should not match price
        ]
        
        for text, expected_hashtags in test_cases:
            hashtags = CommentValidator.extract_hashtags(text)
            assert set(hashtags) == set(expected_hashtags)
