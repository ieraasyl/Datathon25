import re
from typing import List, Dict, Optional
import logging
from ..models.comments import EnhancedComment

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Basic sentiment analysis (can be enhanced with ML models)"""
    
    POSITIVE_WORDS = [
        'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
        'love', 'like', 'best', 'perfect', 'brilliant', 'outstanding', 'superb',
        'beautiful', 'incredible', 'magnificent', 'marvelous', 'spectacular'
    ]
    
    NEGATIVE_WORDS = [
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
        'worst', 'pathetic', 'useless', 'stupid', 'ridiculous', 'disappointing',
        'annoying', 'frustrating', 'boring', 'waste', 'fail', 'sucks'
    ]
    
    @classmethod
    def analyze_sentiment(cls, text: str) -> str:
        """Simple rule-based sentiment analysis"""
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        positive_score = sum(1 for word in cls.POSITIVE_WORDS if word in text_lower)
        negative_score = sum(1 for word in cls.NEGATIVE_WORDS if word in text_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'

class ContentAnalyzer:
    """Analyze comment content for additional insights"""
    
    @staticmethod
    def contains_questions(text: str) -> bool:
        """Check if comment contains questions"""
        if not text:
            return False
        return '?' in text or any(
            text.lower().startswith(word) for word in 
            ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'do', 'does', 'did', 'can', 'could', 'would', 'should']
        )
    
    @staticmethod
    def contains_urls(text: str) -> bool:
        """Check if comment contains URLs"""
        if not text:
            return False
        return bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    
    @staticmethod
    def count_emojis(text: str) -> int:
        """Count emoji characters (basic implementation)"""
        if not text:
            return 0
        # Basic emoji patterns - can be enhanced with emoji library
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # emoticons
            r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]|'  # transport & map symbols
            r'[\U0001F1E0-\U0001F1FF]'   # flags
        )
        return len(emoji_pattern.findall(text))
    
    @staticmethod
    def is_spam_like(text: str) -> bool:
        """Basic spam detection"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for excessive repetition
        words = text_lower.split()
        if len(set(words)) < len(words) * 0.5 and len(words) > 3:
            return True
        
        # Check for excessive caps
        if len(text) > 10 and sum(1 for c in text if c.isupper()) > len(text) * 0.7:
            return True
        
        # Check for spam keywords
        spam_keywords = ['buy now', 'click here', 'free money', 'viagra', 'casino']
        if any(keyword in text_lower for keyword in spam_keywords):
            return True
        
        return False

class DataEnricher:
    """Enrich comment data with additional analytics"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.content_analyzer = ContentAnalyzer()
    
    def enrich_comment(self, comment: EnhancedComment) -> EnhancedComment:
        """Enrich a single comment with additional data"""
        if comment.text:
            # Add sentiment analysis
            comment.sentiment = self.sentiment_analyzer.analyze_sentiment(comment.text)
            
            # Add content analysis flags
            comment.has_questions = self.content_analyzer.contains_questions(comment.text)
            comment.has_urls = self.content_analyzer.contains_urls(comment.text)
            comment.emoji_count = self.content_analyzer.count_emojis(comment.text)
            comment.is_spam_like = self.content_analyzer.is_spam_like(comment.text)
        
        return comment
    
    def enrich_comments_batch(self, comments: List[EnhancedComment]) -> List[EnhancedComment]:
        """Enrich a batch of comments"""
        enriched_comments = []
        for comment in comments:
            try:
                enriched_comment = self.enrich_comment(comment)
                enriched_comments.append(enriched_comment)
            except Exception as e:
                logger.warning(f"Error enriching comment {comment.id}: {str(e)}")
                enriched_comments.append(comment)  # Add original comment if enrichment fails
        
        return enriched_comments
