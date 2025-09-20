import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from ..models.comments import EnhancedComment

class DataExporter:
    """Export comments to various formats"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def to_json(self, comments: List[EnhancedComment], filename: str) -> str:
        """Export comments to JSON format"""
        filepath = self.output_dir / f"{filename}.json"
        
        data = {
            'metadata': {
                'total_comments': len(comments),
                'export_timestamp': pd.Timestamp.now().isoformat(),
                'format_version': '1.0'
            },
            'comments': [self._comment_to_dict(comment) for comment in comments]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def to_excel(self, comments: List[EnhancedComment], filename: str) -> str:
        """Export comments to Excel format with multiple sheets"""
        filepath = self.output_dir / f"{filename}.xlsx"
        
        # Convert comments to DataFrame
        df = pd.DataFrame([self._comment_to_dict(comment) for comment in comments])
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Comments', index=False)
            
            # Summary sheet
            summary_data = self._generate_summary(comments)
            summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Platform breakdown
            platform_stats = df.groupby('platform').agg({
                'id': 'count',
                'likes_count': ['mean', 'sum'],
                'word_count': 'mean',
                'sentiment': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0
            }).round(2)
            platform_stats.to_excel(writer, sheet_name='Platform Stats')
        
        return str(filepath)
    
    def to_xml(self, comments: List[EnhancedComment], filename: str) -> str:
        """Export comments to XML format"""
        filepath = self.output_dir / f"{filename}.xml"
        
        root = ET.Element('comments')
        root.set('total', str(len(comments)))
        root.set('export_timestamp', pd.Timestamp.now().isoformat())
        
        for comment in comments:
            comment_elem = ET.SubElement(root, 'comment')
            for key, value in self._comment_to_dict(comment).items():
                if value is not None:
                    elem = ET.SubElement(comment_elem, key)
                    elem.text = str(value)
        
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        return str(filepath)
    
    def _comment_to_dict(self, comment: EnhancedComment) -> Dict[str, Any]:
        """Convert comment to dictionary"""
        return {
            'id': comment.id,
            'author': comment.author,
            'author_id': comment.author_id,
            'timestamp': comment.timestamp,
            'text': comment.text,
            'platform': comment.platform,
            'post_id': comment.post_id,
            'post_url': comment.post_url,
            'business_name': comment.business_name,
            'likes_count': comment.likes_count or 0,
            'replies_count': comment.replies_count or 0,
            'is_verified': comment.is_verified,
            'language': comment.language,
            'sentiment': comment.sentiment,
            'word_count': comment.word_count or 0,
            'has_mentions': comment.has_mentions,
            'has_hashtags': comment.has_hashtags,
            'scraped_at': comment.scraped_at
        }
    
    def _generate_summary(self, comments: List[EnhancedComment]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not comments:
            return {}
        
        df = pd.DataFrame([self._comment_to_dict(comment) for comment in comments])
        
        return {
            'total_comments': len(comments),
            'unique_authors': df['author'].nunique(),
            'platforms': df['platform'].nunique(),
            'avg_likes_per_comment': df['likes_count'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'comments_with_mentions': df['has_mentions'].sum(),
            'comments_with_hashtags': df['has_hashtags'].sum(),
            'positive_sentiment_ratio': (df['sentiment'] == 'positive').sum() / len(df) if len(df) > 0 else 0,
            'negative_sentiment_ratio': (df['sentiment'] == 'negative').sum() / len(df) if len(df) > 0 else 0
        }