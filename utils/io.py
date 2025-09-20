#!/usr/bin/env python3
"""
I/O utilities with retry logic and error handling
Author: Yerassyl
"""

import json
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

from constants import ERROR_MESSAGES, SUCCESS_MESSAGES, RETRY_CONFIG
from models import ValidationResult, validate_comments, validate_classified, validate_replies
from utils.logging import get_logger

logger = get_logger(__name__)

class FileIOManager:
    """Centralized file I/O operations with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retry_attempts = config.get('pipeline', {}).get('retry_attempts', RETRY_CONFIG['MAX_ATTEMPTS'])
        self.retry_delay = config.get('pipeline', {}).get('retry_delay', RETRY_CONFIG['BASE_DELAY'])
    
    @retry(
        stop=stop_after_attempt(RETRY_CONFIG['MAX_ATTEMPTS']),
        wait=wait_exponential(multiplier=RETRY_CONFIG['BASE_DELAY'], max=RETRY_CONFIG['MAX_DELAY']),
        retry=retry_if_exception_type((FileNotFoundError, PermissionError, OSError))
    )
    def load_json(self, file_path: Path, validate_func=None) -> Tuple[List[dict], ValidationResult]:
        """Load and validate JSON data with retry logic"""
        try:
            logger.info(f"Loading JSON from {file_path}")
            
            if not file_path.exists():
                logger.warning(ERROR_MESSAGES['FILE_NOT_FOUND'].format(file_path=file_path))
                return [], ValidationResult(is_valid=True, valid_records=0, invalid_records=0)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data] if isinstance(data, dict) else []
            
            # Validate data if validation function provided
            if validate_func and data:
                validation_result, valid_data = validate_func(data)
                if not validation_result.is_valid:
                    logger.warning(f"Validation issues in {file_path}: {validation_result.errors}")
                data = valid_data
            else:
                validation_result = ValidationResult(is_valid=True, valid_records=len(data), invalid_records=0)
            
            logger.info(SUCCESS_MESSAGES['DATA_LOADED'].format(count=len(data), file_path=file_path))
            return data, validation_result
            
        except json.JSONDecodeError as e:
            logger.error(ERROR_MESSAGES['INVALID_JSON'].format(file_path=file_path))
            raise
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(RETRY_CONFIG['MAX_ATTEMPTS']),
        wait=wait_exponential(multiplier=RETRY_CONFIG['BASE_DELAY'], max=RETRY_CONFIG['MAX_DELAY']),
        retry=retry_if_exception_type((PermissionError, OSError))
    )
    def save_json(self, data: Any, file_path: Path) -> bool:
        """Save data to JSON with retry logic"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"JSON saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(RETRY_CONFIG['MAX_ATTEMPTS']),
        wait=wait_exponential(multiplier=RETRY_CONFIG['BASE_DELAY'], max=RETRY_CONFIG['MAX_DELAY']),
        retry=retry_if_exception_type((PermissionError, OSError))
    )
    def save_csv(self, df: pd.DataFrame, file_path: Path, **kwargs) -> bool:
        """Save DataFrame to CSV with retry logic"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Default CSV settings
            csv_kwargs = {
                'index': False,
                'encoding': 'utf-8'
            }
            csv_kwargs.update(kwargs)
            
            df.to_csv(file_path, **csv_kwargs)
            logger.info(SUCCESS_MESSAGES['FILE_EXPORTED'].format(format='CSV', file_path=file_path))
            return True
            
        except Exception as e:
            logger.error(ERROR_MESSAGES['EXPORT_FAILED'].format(format='CSV', error=e))
            raise
    
    @retry(
        stop=stop_after_attempt(RETRY_CONFIG['MAX_ATTEMPTS']),
        wait=wait_exponential(multiplier=RETRY_CONFIG['BASE_DELAY'], max=RETRY_CONFIG['MAX_DELAY']),
        retry=retry_if_exception_type((PermissionError, OSError))
    )
    def save_excel(self, df: pd.DataFrame, file_path: Path, include_summary: bool = True) -> bool:
        """Save DataFrame to Excel with summary sheet"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Comments_Analysis', index=False)
                
                # Summary sheet
                if include_summary:
                    summary_data = self._create_summary_data(df)
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # QA sheet
                    qa_data = self._create_qa_data(df)
                    qa_df = pd.DataFrame(qa_data)
                    qa_df.to_excel(writer, sheet_name='Quality_Analysis', index=False)
            
            logger.info(SUCCESS_MESSAGES['FILE_EXPORTED'].format(format='Excel', file_path=file_path))
            return True
            
        except Exception as e:
            logger.error(ERROR_MESSAGES['EXPORT_FAILED'].format(format='Excel', error=e))
            raise
    
    def _create_summary_data(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create summary statistics for Excel export"""
        return {
            'Metric': [
                'Total Comments',
                'Unique Authors', 
                'Languages Detected',
                'Sentiment Categories',
                'Comment Categories',
                'Comments with Replies',
                'Average Text Length',
                'Date Range'
            ],
            'Value': [
                len(df),
                df['author'].nunique() if 'author' in df.columns else 0,
                df['lang'].nunique() if 'lang' in df.columns else 0,
                df['sentiment'].nunique() if 'sentiment' in df.columns else 0,
                df['category'].nunique() if 'category' in df.columns else 0,
                df['reply'].notna().sum() if 'reply' in df.columns else 0,
                f"{df['text'].str.len().mean():.1f}" if 'text' in df.columns else 0,
                f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A'
            ]
        }
    
    def _create_qa_data(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create quality analysis data for Excel export"""
        qa_metrics = []
        
        for column in df.columns:
            null_count = df[column].isna().sum()
            null_percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0
            
            qa_metrics.append({
                'Column': column,
                'Null_Count': null_count,
                'Null_Percentage': f"{null_percentage:.2f}%",
                'Data_Type': str(df[column].dtype),
                'Unique_Values': df[column].nunique(),
                'Quality_Flag': 'WARN' if null_percentage > 25 else 'OK'
            })
        
        return {key: [item[key] for item in qa_metrics] for key in qa_metrics[0].keys()}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return _get_default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return _get_default_config()

def _get_default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        'data': {
            'base_path': './',
            'raw_path': 'data/raw',
            'processed_path': 'data/processed', 
            'output_path': 'output'
        },
        'input_files': {
            'comments': 'comments.json',
            'classified': 'classified.json',
            'replies': 'replies.json'
        },
        'output_files': {
            'csv': 'final.csv',
            'excel': 'final.xlsx',
            'pdf': 'final.pdf',
            'metadata': 'metadata.json'
        },
        'pipeline': {
            'create_sample_data': True,
            'preservation_rate_threshold': 90.0,
            'retry_attempts': 3
        }
    }