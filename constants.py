#!/usr/bin/env python3
"""
Constants and configuration mappings for Social Media ETL Pipeline
Author: Yerassyl
"""

# Column mappings for data consistency
COLUMNS = {
    'ID': 'id',
    'TEXT': 'text',
    'AUTHOR': 'username',  # Updated from 'author' to 'username'
    'TIMESTAMP': 'created_at_utc',  # Updated from 'timestamp'
    'REPLY': 'reply',
    'LANG': 'lang',
    'CATEGORY': 'category',
    'SENTIMENT': 'sentiment',
    'MODERATION': 'moderation'
}

# Default fill values for missing data
FILL_VALUES = {
    'lang': 'unknown',
    'moderation': 'safe',
    'category': 'unknown',
    'sentiment': 'neutral',
    'reply': None,
    'username': 'unknown_user',
    'like_count': 0
}

# Valid values for categorical fields
VALID_LANGUAGES = ['en', 'ru', 'kk', 'mixed', 'unknown', 'other']
VALID_SENTIMENTS = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
VALID_CATEGORIES = ['complaint', 'thanks', 'question', 'review', 'suggestion', 'spam', 'other', 'unknown']
VALID_MODERATION = ['safe', 'offensive', 'spam', 'unknown', 'flagged']

# Gemini AI mappings
GEMINI_TYPE_TO_CATEGORY = {
    'complaint': 'complaint',
    'thanks': 'thanks', 
    'question': 'question',
    'review': 'review',
    'spam': 'spam'
}

GEMINI_SENTIMENT_TO_STANDARD = {
    'very_positive': 'positive',
    'positive': 'positive',
    'neutral': 'neutral',
    'negative': 'negative',
    'very_negative': 'negative'
}

# Social media file patterns
SOCIAL_MEDIA_PATTERNS = [
    'altel_*.json',
    'tele2_*.json'
]

# Error messages
ERROR_MESSAGES = {
    'PIPELINE_FAILED': "Pipeline execution failed: {error}",
    'EXPORT_FAILED': "Failed to export {format}: {error}",
    'LOW_PRESERVATION_RATE': "Low data preservation rate: {rate:.1f}% < {threshold}%",
    'GEMINI_API_ERROR': "Gemini API error: {error}",
    'FILE_NOT_FOUND': "File not found: {file_path}",
    'VALIDATION_ERROR': "Data validation failed: {error}"
}

# Success messages
SUCCESS_MESSAGES = {
    'PIPELINE_COMPLETED': "ETL Pipeline completed successfully in {time:.2f} seconds",
    'FILE_EXPORTED': "{format} file exported successfully: {file_path}",
    'GEMINI_ANALYSIS_COMPLETE': "Gemini analysis completed: {success_count}/{total_count} successful",
    'DATA_EXTRACTED': "Successfully extracted {count} records from {source}",
    'VALIDATION_PASSED': "Data validation passed: {valid_count} valid records"
}

# Configuration defaults
DEFAULT_CONFIG = {
    'data': {
        'base_path': '.',
        'raw_path': 'data/raw',
        'processed_path': 'data/processed', 
        'output_path': 'data/output',
        'logs_path': 'logs'
    },
    'gemini': {
        'enabled': True,
        'batch_size': 50,
        'delay_between_requests': 1.0,
        'max_retries': 3
    },
    'pipeline': {
        'preservation_rate_threshold': 85.0,
        'max_pdf_rows': 100,
        'max_text_length': 100
    },
    'logging': {
        'level': 'INFO',
        'file_logging': True,
        'console_logging': True
    }
}