#!/usr/bin/env python3
"""
Logging utilities for ETL Pipeline
Author: Yerassyl
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
from functools import wraps
import time

def setup_logging(config_path: str = "config.yaml"):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.INFO
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(logs_dir / "etl_pipeline.log", encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress verbose third-party logs
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

def timed_operation(logger: logging.Logger, operation_name: str):
    """Decorator to time operations and log results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {operation_name}...")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"✅ {operation_name} completed in {elapsed_time:.2f} seconds")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"❌ {operation_name} failed after {elapsed_time:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator