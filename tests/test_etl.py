#!/usr/bin/env python3
"""
Basic smoke tests for ETL pipeline
Author: Yerassyl
"""

import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from constants import COLUMNS, FILL_VALUES
from models import validate_comments, validate_classified, validate_replies
from utils.io import FileIOManager, load_config
from etl.main import ETLPipeline

class TestETLPipeline:
    """Basic smoke tests for ETL pipeline functionality"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config for testing"""
        config = {
            'data': {
                'base_path': str(Path.cwd()),
                'raw_path': 'test_data/raw',
                'processed_path': 'test_data/processed',
                'output_path': 'test_output'
            },
            'input_files': {
                'comments': 'test_comments.json',
                'classified': 'test_classified.json',
                'replies': 'test_replies.json'
            },
            'output_files': {
                'csv': 'test_final.csv',
                'excel': 'test_final.xlsx',
                'pdf': 'test_final.pdf',
                'metadata': 'test_metadata.json'
            },
            'pipeline': {
                'create_sample_data': False,
                'preservation_rate_threshold': 90.0,
                'max_pdf_rows': 10,
                'retry_attempts': 1
            }
        }
        return config
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        comments = [
            {"id": "test1", "author": "user1", "timestamp": "2025-09-20T10:00:00", "text": "Test comment 1"},
            {"id": "test2", "author": "user2", "timestamp": "2025-09-20T10:05:00", "text": "Test comment 2"}
        ]
        
        classified = [
            {"id": "test1", "lang": "en", "moderation": "safe", "category": "test", "sentiment": "neutral"},
            {"id": "test2", "lang": "en", "moderation": "safe", "category": "test", "sentiment": "positive"}
        ]
        
        replies = [
            {"id": "test1", "reply": "Test reply 1"},
            {"id": "test2", "reply": None}
        ]
        
        return comments, classified, replies
    
    def test_load_config(self):
        """Test configuration loading"""
        # Test default config when file doesn't exist
        config = load_config("nonexistent.yaml")
        assert isinstance(config, dict)
        assert 'data' in config
        assert 'input_files' in config
    
    def test_validate_comments(self, sample_data):
        """Test comment validation"""
        comments, _, _ = sample_data
        
        validation_result, valid_records = validate_comments(comments)
        
        assert validation_result.is_valid
        assert validation_result.valid_records == 2
        assert validation_result.invalid_records == 0
        assert len(valid_records) == 2
    
    def test_validate_classified(self, sample_data):
        """Test classified data validation"""
        _, classified, _ = sample_data
        
        validation_result, valid_records = validate_classified(classified)
        
        assert validation_result.is_valid
        assert validation_result.valid_records == 2
        assert validation_result.invalid_records == 0
        assert len(valid_records) == 2
    
    def test_validate_replies(self, sample_data):
        """Test reply validation"""
        _, _, replies = sample_data
        
        validation_result, valid_records = validate_replies(replies)
        
        assert validation_result.is_valid
        assert validation_result.valid_records == 2
        assert validation_result.invalid_records == 0
        assert len(valid_records) == 2
        assert valid_records[1]['reply'] is None  # Test null handling
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        invalid_comments = [
            {"id": "", "text": "", "timestamp": "invalid"},  # Empty ID and text, invalid timestamp
            {"id": "valid1", "author": "user", "timestamp": "2025-09-20T10:00:00", "text": "Valid comment"}
        ]
        
        validation_result, valid_records = validate_comments(invalid_comments)
        
        assert not validation_result.is_valid
        assert validation_result.valid_records == 1  # Only one valid record
        assert validation_result.invalid_records == 1
        assert len(validation_result.errors) > 0
    
    def test_file_io_manager(self, temp_config, sample_data):
        """Test FileIOManager basic functionality"""
        comments, classified, replies = sample_data
        io_manager = FileIOManager(temp_config)
        
        # Test JSON save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test.json"
            
            # Save
            success = io_manager.save_json(comments, test_path)
            assert success
            assert test_path.exists()
            
            # Load
            loaded_data, validation = io_manager.load_json(test_path, validate_comments)
            assert len(loaded_data) == 2
            assert validation.is_valid
    
    def test_csv_export(self, temp_config):
        """Test CSV export functionality"""
        test_df = pd.DataFrame({
            COLUMNS['ID']: ['test1', 'test2'],
            COLUMNS['TEXT']: ['Test 1', 'Test 2'],
            COLUMNS['SENTIMENT']: ['positive', 'negative']
        })
        
        io_manager = FileIOManager(temp_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            
            success = io_manager.save_csv(test_df, csv_path)
            assert success
            assert csv_path.exists()
            
            # Verify content
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == 2
            assert COLUMNS['ID'] in loaded_df.columns
    
    def test_data_merge_logic(self, sample_data):
        """Test the core data merging logic"""
        comments, classified, replies = sample_data
        
        # Convert to DataFrames
        df_comments = pd.DataFrame(comments)
        df_classified = pd.DataFrame(classified)
        df_replies = pd.DataFrame(replies)
        
        # Test merge logic (similar to ETLPipeline.transform_data)
        merged_df = df_comments.copy()
        merged_df = merged_df.merge(df_classified, on=COLUMNS['ID'], how='left')
        merged_df = merged_df.merge(df_replies, on=COLUMNS['ID'], how='left')
        
        # Fill missing values
        for col, fill_val in FILL_VALUES.items():
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(fill_val)
        
        # Verify merge results
        assert len(merged_df) == 2
        assert COLUMNS['LANGUAGE'] in merged_df.columns
        assert COLUMNS['SENTIMENT'] in merged_df.columns
        assert COLUMNS['REPLY'] in merged_df.columns
        
        # Check data preservation
        initial_count = len(df_comments)
        final_count = len(merged_df)
        preservation_rate = (final_count / initial_count * 100)
        assert preservation_rate >= 90.0  # Meet threshold
    
    def test_pipeline_initialization(self):
        """Test ETL pipeline initialization"""
        # Test with default config
        pipeline = ETLPipeline()
        assert pipeline.run_id is not None
        assert pipeline.logger is not None
        assert hasattr(pipeline, 'paths')
        assert hasattr(pipeline, 'io_manager')
    
    @pytest.mark.integration
    def test_full_pipeline_sample_data(self, temp_config):
        """Integration test with sample data creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config to use temp directory
            temp_config['data']['base_path'] = temp_dir
            
            # Create config file
            import yaml
            config_path = Path(temp_dir) / "test_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(temp_config, f)
            
            # Initialize pipeline
            pipeline = ETLPipeline(str(config_path))
            
            # Run with sample data
            success = pipeline.run_pipeline(create_sample=True)
            
            # Verify success and outputs
            assert success
            
            # Check if output files were created
            csv_path = Path(temp_dir) / temp_config['data']['output_path'] / temp_config['output_files']['csv']
            assert csv_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(csv_path)
            assert len(df) > 0
            assert COLUMNS['ID'] in df.columns
            assert COLUMNS['TEXT'] in df.columns

class TestDataValidation:
    """Tests specifically for data validation logic"""
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        invalid_comment = [{"id": "test1", "text": "Missing timestamp"}]  # Missing timestamp
        
        validation_result, valid_records = validate_comments(invalid_comment)
        assert not validation_result.is_valid
        assert validation_result.invalid_records == 1
    
    def test_edge_cases(self):
        """Test edge cases in validation"""
        edge_cases = [
            {"id": "test1", "author": None, "timestamp": "2025-09-20T10:00:00", "text": "   "},  # Whitespace text
            {"id": "test2", "author": "", "timestamp": "2025-09-20T10:00:00.123", "text": "Valid text"}  # Microseconds
        ]
        
        validation_result, valid_records = validate_comments(edge_cases)
        # First should fail (whitespace only text), second should pass
        assert validation_result.invalid_records == 1
        assert validation_result.valid_records == 1

def run_smoke_tests():
    """Run basic smoke tests programmatically"""
    print("Running ETL Pipeline smoke tests...")
    
    # Test basic imports
    try:
        from constants import COLUMNS
        from models import Comment
        from utils.io import load_config
        print("✅ Imports working")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test basic validation
    try:
        test_comment = {"id": "test", "author": "user", "timestamp": "2025-09-20T10:00:00", "text": "Test"}
        validation_result, _ = validate_comments([test_comment])
        assert validation_result.is_valid
        print("✅ Validation working")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Test configuration loading
    try:
        config = load_config("nonexistent.yaml")
        assert isinstance(config, dict)
        print("✅ Configuration loading working")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    print("✅ All smoke tests passed!")
    return True

if __name__ == "__main__":
    run_smoke_tests()