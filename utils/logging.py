#!/usr/bin/env python3
"""
Centralized logging configuration
Author: Yerassyl
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import yaml

class LoggerSetup:
    """Centralized logger configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load logging configuration"""
        default_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_handler': True,
            'console_handler': True,
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    return full_config.get('logging', default_config)
            except Exception:
                pass
        
        return default_config
    
    def _setup_logging(self):
        """Configure logging handlers"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level'].upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(self.config['format'])
        
        # Console handler
        if self.config.get('console_handler', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config['level'].upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.get('file_handler', True):
            file_handler = logging.handlers.RotatingFileHandler(
                logs_dir / "etl.log",
                maxBytes=self.config.get('max_bytes', 10485760),
                backupCount=self.config.get('backup_count', 5),
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.config['level'].upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get configured logger for module"""
        return logging.getLogger(name)

# Global logger setup instance
_logger_setup = None

def setup_logging(config_path: Optional[str] = None) -> None:
    """Initialize logging configuration"""
    global _logger_setup
    _logger_setup = LoggerSetup(config_path)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    global _logger_setup
    if _logger_setup is None:
        setup_logging()
    assert _logger_setup is not None  # Type checker assertion
    return _logger_setup.get_logger(name)

# Convenience function for timed operations
class TimedLogger:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, operation_name: str, level: int = logging.INFO):
        self.logger = logger
        self.operation_name = operation_name
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.start_time:
            duration = time.time() - self.start_time
            if exc_type is None:
                self.logger.log(self.level, f"Completed {self.operation_name} in {duration:.2f} seconds")
            else:
                self.logger.error(f"Failed {self.operation_name} after {duration:.2f} seconds: {exc_val}")

def timed_operation(logger: logging.Logger, operation_name: str, level: int = logging.INFO):
    """Decorator for timing function execution"""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.log(level, f"Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log(level, f"Completed {operation_name} in {duration:.2f} seconds")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation_name} after {duration:.2f} seconds: {e}")
                raise
        return wrapper
    return decorator