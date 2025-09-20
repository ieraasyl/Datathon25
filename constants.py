#!/usr/bin/env python3
"""
Constants for Social Media Analysis Pipeline
Author: Yerassyl
"""

# Data schema constants
REQUIRED_COMMENT_FIELDS = ['id', 'author', 'timestamp', 'text']
REQUIRED_CLASSIFIED_FIELDS = ['id', 'lang', 'moderation', 'category', 'sentiment']
REQUIRED_REPLY_FIELDS = ['id', 'reply']

# Column names
COLUMNS = {
    'ID': 'id',
    'AUTHOR': 'author',
    'TIMESTAMP': 'timestamp',
    'TEXT': 'text',
    'LANGUAGE': 'lang',
    'MODERATION': 'moderation',
    'CATEGORY': 'category',
    'SENTIMENT': 'sentiment',
    'REPLY': 'reply'
}

# Default fill values for missing data
FILL_VALUES = {
    'lang': 'unknown',
    'moderation': 'unknown',
    'category': 'unknown',
    'sentiment': 'unknown',
    'reply': ''
}

# Sentiment values and colors
SENTIMENTS = {
    'positive': '#00CC96',
    'negative': '#EF553B', 
    'neutral': '#636EFA',
    'unknown': '#AB63FA'
}

# Category colors
CATEGORY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

# Language codes mapping
LANGUAGE_MAPPING = {
    'en': 'English',
    'ru': 'Russian',
    'kk': 'Kazakh',
    'mixed': 'Mixed',
    'unknown': 'Unknown'
}

# Moderation status colors
MODERATION_COLORS = {
    'safe': '#00CC96',
    'offensive': '#EF553B',
    'spam': '#FFA15A',
    'unknown': '#636EFA'
}

# UI constants
UI_CONSTANTS = {
    'MAX_TEXT_DISPLAY': 100,
    'MAX_REPLY_DISPLAY': 100,
    'PAGE_SIZE': 50,
    'CHART_HEIGHT': 400,
    'TABLE_HEIGHT': 600
}

# File formats
SUPPORTED_FORMATS = ['csv', 'xlsx', 'pdf', 'json']

# Quality thresholds
QUALITY_THRESHOLDS = {
    'MIN_PRESERVATION_RATE': 90.0,
    'MAX_NULL_PERCENTAGE': 50.0,
    'MAX_DUPLICATE_PERCENTAGE': 5.0,
    'MIN_TEXT_LENGTH': 1,
    'MAX_PROCESSING_TIME': 300  # 5 minutes
}

# Error messages
ERROR_MESSAGES = {
    'FILE_NOT_FOUND': "Input file not found: {file_path}",
    'INVALID_JSON': "Invalid JSON format in file: {file_path}",
    'SCHEMA_VALIDATION_FAILED': "Schema validation failed for {model_name}: {errors}",
    'LOW_PRESERVATION_RATE': "Data preservation rate ({rate:.1f}%) below threshold ({threshold:.1f}%)",
    'EXPORT_FAILED': "Failed to export {format} file: {error}",
    'PIPELINE_FAILED': "ETL pipeline failed: {error}"
}

# Success messages
SUCCESS_MESSAGES = {
    'DATA_LOADED': "Successfully loaded {count} records from {file_path}",
    'PIPELINE_COMPLETED': "ETL Pipeline completed successfully in {time:.2f} seconds",
    'FILE_EXPORTED': "{format} exported: {file_path}",
    'VALIDATION_PASSED': "Input validation passed for {model_name}"
}

# Log formats
LOG_FORMATS = {
    'SIMPLE': '%(asctime)s - %(levelname)s - %(message)s',
    'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'JSON': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
}

# Retry configuration
RETRY_CONFIG = {
    'MAX_ATTEMPTS': 3,
    'BASE_DELAY': 1.0,
    'EXPONENTIAL_BASE': 2,
    'MAX_DELAY': 60.0
}

# Dashboard metrics
METRIC_LABELS = {
    'TOTAL_COMMENTS': 'Total Comments',
    'LANGUAGES_DETECTED': 'Languages Detected', 
    'REPLY_RATE': 'Reply Rate',
    'POSITIVE_SENTIMENT': 'Positive Sentiment',
    'PROCESSING_TIME': 'Processing Time',
    'PRESERVATION_RATE': 'Data Preservation Rate'
}

# Export settings
EXPORT_SETTINGS = {
    'CSV': {
        'encoding': 'utf-8',
        'index': False
    },
    'EXCEL': {
        'engine': 'openpyxl',
        'index': False
    },
    'PDF': {
        'pagesize': 'A4',
        'max_rows': 50,
        'font_size': 6
    }
}