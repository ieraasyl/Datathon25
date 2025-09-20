import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
import re
from ..models.comments import EnhancedComment

class CSVHandler:
    """Handle CSV export functionality"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_comments(self, comments: List[EnhancedComment], business_name: str, export_format: str = "detailed") -> str:
        """Save comments to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_business_name = re.sub(r'[^\w\-_\. ]', '_', business_name)
        filename = f"comments_{safe_business_name}_{timestamp}_{uuid.uuid4().hex[:6]}.csv"
        filepath = self.output_dir / filename
        
        fieldnames = self._get_fieldnames(export_format)
        
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for comment in comments:
                row_data = {}
                for field in fieldnames:
                    row_data[field] = getattr(comment, field, '')
                writer.writerow(row_data)
        
        return str(filepath)
    
    def _get_fieldnames(self, export_format: str) -> List[str]:
        """Get CSV fieldnames based on export format"""
        if export_format == "detailed":
            return [
                'id', 'author', 'author_id', 'timestamp', 'text', 'platform', 
                'post_id', 'post_url', 'business_name', 'likes_count', 'replies_count',
                'is_verified', 'language', 'sentiment', 'word_count', 'has_mentions',
                'has_hashtags', 'scraped_at'
            ]
        else:  # simple format
            return ['id', 'author', 'timestamp', 'text', 'platform', 'post_url', 'business_name']
