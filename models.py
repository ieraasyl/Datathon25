#!/usr/bin/env python3
"""
Pydantic models for input validation
Author: Yerassyl
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
from datetime import datetime
import re

class Comment(BaseModel):
    """Model for raw comments from parser (Daulet)"""
    id: str = Field(..., min_length=1, description="Unique comment identifier")
    author: Optional[str] = Field(None, description="Comment author username")
    timestamp: str = Field(..., description="Comment timestamp in ISO format")
    text: str = Field(..., min_length=1, description="Comment text content")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            # Try to parse various common timestamp formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']:
                try:
                    datetime.strptime(v, fmt)
                    return v
                except ValueError:
                    continue
            raise ValueError("Invalid timestamp format")
        except Exception:
            raise ValueError(f"Invalid timestamp: {v}")
    
    @validator('text')
    def validate_text_length(cls, v):
        """Ensure text is not just whitespace"""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

class Classified(BaseModel):
    """Model for classified data from ML pipeline (Olzhas)"""
    id: str = Field(..., min_length=1, description="Comment identifier matching Comment.id")
    lang: str = Field(..., description="Detected language code")
    moderation: str = Field(..., description="Moderation status")
    category: str = Field(..., description="Comment category")
    sentiment: str = Field(..., description="Sentiment analysis result")
    
    @validator('lang')
    def validate_language(cls, v):
        """Validate language code"""
        valid_langs = ['en', 'ru', 'kk', 'mixed', 'unknown', 'other']
        if v.lower() not in valid_langs:
            # Don't fail validation, just normalize
            return 'unknown'
        return v.lower()
    
    @validator('moderation')
    def validate_moderation(cls, v):
        """Validate moderation status"""
        valid_statuses = ['safe', 'offensive', 'spam', 'unknown', 'flagged']
        if v.lower() not in valid_statuses:
            return 'unknown'
        return v.lower()
    
    @validator('category')
    def validate_category(cls, v):
        """Validate category"""
        valid_categories = [
            'complaint', 'thanks', 'question', 'review', 'suggestion', 
            'bug_report', 'feature_request', 'other', 'unknown'
        ]
        if v.lower() not in valid_categories:
            return 'unknown'
        return v.lower()
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Validate sentiment"""
        valid_sentiments = ['positive', 'negative', 'neutral', 'unknown']
        if v.lower() not in valid_sentiments:
            return 'unknown'
        return v.lower()

class Reply(BaseModel):
    """Model for generated replies (Anuar)"""
    id: str = Field(..., min_length=1, description="Comment identifier matching Comment.id")
    reply: Optional[str] = Field(None, description="Generated reply text or null if no reply")
    
    @validator('reply', pre=True)
    def validate_reply(cls, v):
        """Handle null/empty replies"""
        if v is None or v == '' or str(v).lower() == 'null':
            return None
        return str(v).strip() if str(v).strip() else None

class MergedComment(BaseModel):
    """Model for final merged dataset"""
    id: str
    author: Optional[str] = None
    timestamp: str
    text: str
    lang: str = 'unknown'
    moderation: str = 'unknown'
    category: str = 'unknown'
    sentiment: str = 'unknown'
    reply: Optional[str] = None

class PipelineMetadata(BaseModel):
    """Model for pipeline execution metadata"""
    pipeline_run_id: str
    start_time: datetime
    end_time: datetime
    processing_time_seconds: float
    total_comments: int
    successful_merges: int
    preservation_rate: float
    input_files: dict
    output_files: dict
    schema_version: str = "1.0"
    
class ValidationResult(BaseModel):
    """Model for validation results"""
    is_valid: bool
    valid_records: int
    invalid_records: int
    errors: List[str] = []
    warnings: List[str] = []

def validate_comments(data: List[dict]) -> ValidationResult:
    """Validate list of comments"""
    valid_records = []
    errors = []
    warnings = []
    
    for i, item in enumerate(data):
        try:
            comment = Comment(**item)
            valid_records.append(comment.dict())
        except Exception as e:
            errors.append(f"Row {i}: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        valid_records=len(valid_records),
        invalid_records=len(errors),
        errors=errors,
        warnings=warnings
    ), valid_records

def validate_classified(data: List[dict]) -> ValidationResult:
    """Validate list of classified data"""
    valid_records = []
    errors = []
    warnings = []
    
    for i, item in enumerate(data):
        try:
            classified = Classified(**item)
            valid_records.append(classified.dict())
        except Exception as e:
            errors.append(f"Row {i}: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        valid_records=len(valid_records),
        invalid_records=len(errors),
        errors=errors,
        warnings=warnings
    ), valid_records

def validate_replies(data: List[dict]) -> ValidationResult:
    """Validate list of replies"""
    valid_records = []
    errors = []
    warnings = []
    
    for i, item in enumerate(data):
        try:
            reply = Reply(**item)
            valid_records.append(reply.dict())
        except Exception as e:
            errors.append(f"Row {i}: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        valid_records=len(valid_records),
        invalid_records=len(errors),
        errors=errors,
        warnings=warnings
    ), valid_records