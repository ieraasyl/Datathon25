#!/usr/bin/env python3
"""
File I/O utilities for ETL Pipeline
Author: Yerassyl
"""

import json
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
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
                'delay_between_requests': 1.0
            },
            'pipeline': {
                'preservation_rate_threshold': 85.0,
                'max_pdf_rows': 100,
                'max_text_length': 100
            },
            'input_files': {
                'comments': 'comments.json',
                'classified': 'classified.json', 
                'replies': 'replies.json'
            },
            'output_files': {
                'csv': 'social_media_analysis.csv',
                'excel': 'social_media_analysis.xlsx',
                'pdf': 'social_media_report.pdf',
                'metadata': 'pipeline_metadata.json'
            }
        }
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

class FileIOManager:
    """Handles file I/O operations for the pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_json(self, data: Any, file_path: Path) -> bool:
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"JSON saved: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {e}")
            return False
    
    def load_json(self, file_path: Path, validator: Optional[Callable] = None) -> Tuple[Any, Any]:
        """Load and validate JSON data"""
        try:
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                return None, None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if validator:
                validation_result, validated_data = validator(data)
                self.logger.info(f"JSON loaded and validated: {file_path}")
                return validated_data, validation_result
            else:
                self.logger.info(f"JSON loaded: {file_path}")
                return data, None
                
        except Exception as e:
            self.logger.error(f"Error loading JSON from {file_path}: {e}")
            return None, None
    
    def save_csv(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Save DataFrame to CSV"""
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            self.logger.info(f"CSV saved: {file_path} ({len(df)} rows)")
            return True
        except Exception as e:
            self.logger.error(f"Error saving CSV to {file_path}: {e}")
            return False
    
    def save_excel(self, df: pd.DataFrame, file_path: Path, include_summary: bool = False) -> bool:
        """Save DataFrame to Excel with optional summary sheet"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Comments', index=False)
                
                if include_summary:
                    # Create summary sheet
                    summary_data = {
                        'Metric': [
                            'Total Comments',
                            'Unique Users', 
                            'Average Likes',
                            'Languages Detected',
                            'Sentiment Distribution'
                        ],
                        'Value': [
                            len(df),
                            df['username'].nunique() if 'username' in df.columns else 'N/A',
                            df['like_count'].mean() if 'like_count' in df.columns else 'N/A',
                            df['lang'].nunique() if 'lang' in df.columns else 'N/A',
                            df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else 'N/A'
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"Excel saved: {file_path} ({len(df)} rows)")
            return True
        except Exception as e:
            self.logger.error(f"Error saving Excel to {file_path}: {e}")
            return False