import pytest
from unittest.mock import Mock, patch
import requests
from src.social_scraper.scrapers.facebook import FacebookScraper
from src.social_scraper.models.comments import EnhancedComment

class TestFacebookScraper:
    
    def test_extract_post_id_valid_urls(self):
        """Test extracting post ID from Facebook URLs"""
        scraper = FacebookScraper("test_token")
        
        test_cases = [
            ("https://www.facebook.com/posts/123456789", "123456789"),
            ("https://facebook.com/permalink.php?story_fbid=987654321", "987654321"),
            ("https://www.facebook.com/page/posts/456789123", "456789123"),
        ]
        
        for url, expected_id in test_cases:
            assert scraper.extract_post_id(url) == expected_id
    
    def test_extract_post_id_invalid_urls(self):
        """Test handling invalid Facebook URLs"""
        scraper = FacebookScraper("test_token")
        
        invalid_urls = [
            "https://www.youtube.com/watch?v=123",
            "https://facebook.com/profile/user",
            "not-a-url"
        ]
        
        for url in invalid_urls:
            assert scraper.extract_post_id(url) is None
    
    @patch('requests.get')
    async def test_scrape_comments_success(self, mock_get):
        """Test successful Facebook comment scraping"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'data': [{
                'id': 'fb_comment_1',
                'from': {'name': 'Test User', 'id': 'user_123'},
                'message': 'Great post!',
                'created_time': '2024-01-01T12:00:00+0000',
                'like_count': 10,
                'comment_count': 2
            }],
            'paging': {}
        }
        mock_get.return_value = mock_response
        
        scraper = FacebookScraper("test_token")
        comments = await scraper.scrape_comments(
            "https://www.facebook.com/posts/123456789",
            max_comments=10,
            business_name="Test Business"
        )
        
        assert len(comments) == 1
        assert isinstance(comments[0], EnhancedComment)
        assert comments[0].author == "Test User"
        assert comments[0].text == "Great post!"
        assert comments[0].platform == "facebook"

