#!/usr/bin/env python3
"""
ETL Pipeline Main Module with Gemini AI Integration
Author: Yerassyl
Description: Modularized ETL pipeline with CLI interface and Gemini analysis
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import time
import os
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from constants import *
from models import (
    PipelineMetadata, validate_comments, validate_classified, validate_replies,
    analyze_and_classify_comments, AnalysisResult, load_social_media_files
)
from utils.io import FileIOManager, load_config
from utils.logging import setup_logging, get_logger, timed_operation

class ETLPipeline:
    """Modularized ETL Pipeline with validation, Gemini analysis, and error handling"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Setup logging first
        setup_logging(config_path)
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config = load_config(config_path)
        self.io_manager = FileIOManager(self.config)
        
        # Setup paths
        self._setup_paths()
        
        # Pipeline metadata
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = None
        self.metadata = {}
        
        # Gemini analysis settings
        self.enable_gemini = self.config.get('gemini', {}).get('enabled', True)
        self.gemini_batch_size = self.config.get('gemini', {}).get('batch_size', 50)
        
        self.logger.info(f"ETL Pipeline initialized with run ID: {self.run_id}")
        self.logger.info(f"Gemini analysis: {'enabled' if self.enable_gemini else 'disabled'}")
    
    def _setup_paths(self):
        """Setup all required paths from config"""
        base_path = Path(self.config['data']['base_path'])
        
        self.paths = {
            'raw': base_path / self.config['data']['raw_path'],
            'processed': base_path / self.config['data']['processed_path'],
            'output': base_path / self.config['data']['output_path'],
            'logs': base_path / self.config['data'].get('logs_path', 'logs')
        }
        
        # Create directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Directory structure created: {list(self.paths.keys())}")
    
    def find_social_media_files(self):
        """Find all social media JSON files in raw directory"""
        json_files = []
        patterns = ['altel_*.json', 'tele2_*.json']
        
        for pattern in patterns:
            json_files.extend(list(self.paths['raw'].glob(pattern)))
        
        self.logger.info(f"Found {len(json_files)} social media files: {[f.name for f in json_files]}")
        return json_files
    
    @timed_operation(get_logger(__name__), "Data extraction")
    def extract_data(self):
        """Extract and validate data from social media files"""
        self.logger.info("Starting data extraction phase...")
        
        # Find social media files
        social_files = self.find_social_media_files()
        if not social_files:
            self.logger.error("No social media files found! Please check the raw data directory.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Load and process social media files
        comments_data, comments_validation = load_social_media_files(social_files)
        
        # Load classifications with validation (if exists)
        classified_path = self.paths['processed'] / self.config['input_files']['classified']
        classified_data, classified_validation = None, None
        if classified_path.exists():
            classified_data, classified_validation = self.io_manager.load_json(
                classified_path, validate_classified
            )
        
        # Load replies with validation (if exists)
        replies_path = self.paths['processed'] / self.config['input_files']['replies']
        replies_data, replies_validation = None, None
        if replies_path.exists():
            replies_data, replies_validation = self.io_manager.load_json(
                replies_path, validate_replies
            )
        
        # Store validation results in metadata
        self.metadata['validation'] = {
            'comments': comments_validation.model_dump(),
            'classified': classified_validation.model_dump() if classified_validation else None,
            'replies': replies_validation.model_dump() if replies_validation else None
        }
        
        # Store source files info
        self.metadata['source_files'] = [f.name for f in social_files]
        
        # Convert to DataFrames
        df_comments = pd.DataFrame(comments_data) if comments_data else pd.DataFrame()
        df_classified = pd.DataFrame(classified_data) if classified_data else pd.DataFrame()
        df_replies = pd.DataFrame(replies_data) if replies_data else pd.DataFrame()
        
        self.logger.info(f"Extracted: {len(df_comments)} comments from {len(social_files)} files")
        self.logger.info(f"Additional data: {len(df_classified)} classifications, {len(df_replies)} replies")
        
        return df_comments, df_classified, df_replies
    
    @timed_operation(get_logger(__name__), "Gemini AI analysis")
    def analyze_with_gemini(self, comments_data):
        """Analyze comments using Gemini AI"""
        if not self.enable_gemini:
            self.logger.info("Gemini analysis disabled - skipping")
            return [], AnalysisResult(total_processed=len(comments_data))
        
        self.logger.info(f"Starting Gemini analysis for {len(comments_data)} comments...")
        
        def progress_callback(current, total):
            if current % 10 == 0 or current == total:
                self.logger.info(f"Gemini analysis progress: {current}/{total} ({current/total*100:.1f}%)")
        
        # Analyze comments and get classifications
        classified_data, analysis_result = analyze_and_classify_comments(
            comments_data, 
            enable_gemini=True, 
            progress_callback=progress_callback
        )
        
        # Store analysis results in metadata
        self.metadata['gemini_analysis'] = {
            'total_processed': analysis_result.total_processed,
            'success_count': analysis_result.success_count,
            'failed_count': analysis_result.failed_count,
            'success_rate': analysis_result.success_count / analysis_result.total_processed if analysis_result.total_processed > 0 else 0,
            'processing_time': analysis_result.processing_time
        }
        
        # Save classifications to processed directory
        if classified_data:
            classified_path = self.paths['processed'] / 'gemini_classifications.json'
            self.io_manager.save_json(classified_data, classified_path)
            self.logger.info(f"Saved {len(classified_data)} classifications to {classified_path}")
        
        self.logger.info(f"Gemini analysis completed: {analysis_result.success_count}/{analysis_result.total_processed} successful")
        
        return classified_data, analysis_result
    
    @timed_operation(get_logger(__name__), "Data transformation")
    def transform_data(self, df_comments, df_classified, df_replies):
        """Transform and merge data with quality checks"""
        self.logger.info("Starting data transformation phase...")
        
        if df_comments.empty:
            self.logger.warning("No comments data found - cannot proceed with transformation")
            return pd.DataFrame()
        
        initial_count = len(df_comments)
        
        # If no classification data exists, generate it with Gemini
        if df_classified.empty and self.enable_gemini:
            self.logger.info("No existing classifications found - generating with Gemini AI...")
            comments_list = df_comments.to_dict('records')
            classified_data, analysis_result = self.analyze_with_gemini(comments_list)
            
            if classified_data:
                df_classified = pd.DataFrame(classified_data)
                self.logger.info(f"Generated {len(df_classified)} classifications using Gemini")
        
        # Start with comments as base
        merged_df = df_comments.copy()
        
        # Merge classifications
        if not df_classified.empty:
            merged_df = merged_df.merge(df_classified, on='id', how='left')
            self.logger.info(f"Merged {len(df_classified)} classification records")
        else:
            # Add empty classification columns with default values
            classification_defaults = {
                'lang': 'unknown',
                'moderation': 'safe', 
                'category': 'unknown',
                'sentiment': 'neutral'
            }
            for col, default_val in classification_defaults.items():
                merged_df[col] = default_val
            self.logger.warning("No classification data - added default values")
        
        # Merge replies
        if not df_replies.empty:
            merged_df = merged_df.merge(df_replies, on='id', how='left')
            self.logger.info(f"Merged {len(df_replies)} reply records")
        else:
            merged_df['reply'] = None
            self.logger.warning("No reply data - added empty replies")
        
        # Data quality transformations
        merged_df = self._apply_data_quality_fixes(merged_df)
        
        # Calculate preservation rate
        final_count = len(merged_df)
        preservation_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
        
        # Check preservation rate threshold
        threshold = self.config.get('pipeline', {}).get('preservation_rate_threshold', 90)
        if preservation_rate < threshold:
            self.logger.warning(f"Low preservation rate: {preservation_rate:.1f}% < {threshold}%")
        
        self.metadata['transformation'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'preservation_rate': preservation_rate
        }
        
        self.logger.info(f"Data transformation completed. Preservation rate: {preservation_rate:.1f}%")
        return merged_df
    
    def _apply_data_quality_fixes(self, df):
        """Apply data quality fixes and standardization"""
        self.logger.info("Applying data quality fixes...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Convert timestamp to datetime
        if 'created_at_utc' in df.columns:
            df['created_at_utc'] = pd.to_datetime(df['created_at_utc'], errors='coerce')
        
        # Fill missing values with safe approach
        fill_values = {
            'username': 'unknown_user',
            'like_count': 0,
            'lang': 'unknown',
            'moderation': 'safe',
            'category': 'unknown',
            'sentiment': 'neutral',
            'source_file': 'unknown'
        }
        
        # Fill non-null values
        for col, fill_val in fill_values.items():
            if col in df.columns:
                df.loc[:, col] = df[col].fillna(fill_val)
        
        # Handle reply column separately (can be None/null)
        if 'reply' in df.columns:
            # Keep NaN as NaN for replies - this is expected
            pass
        
        # Remove duplicates by ID
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['id'], keep='first')
        after_dedup = len(df)
        
        if before_dedup != after_dedup:
            self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        # Standardize text fields
        text_columns = ['text']
        for col in text_columns:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(str).str.strip()
        
        # Clean username
        if 'username' in df.columns:
            df.loc[:, col] = df['username'].astype(str).str.strip()
        
        # Clean reply column if exists
        if 'reply' in df.columns:
            # Only clean non-null replies
            mask = df['reply'].notna()
            df.loc[mask, 'reply'] = df.loc[mask, 'reply'].astype(str).str.strip()
        
        return df
    
    @timed_operation(get_logger(__name__), "Data export")
    def load_data(self, merged_df):
        """Export data to all required formats"""
        self.logger.info("Starting data export phase...")
        
        if merged_df.empty:
            self.logger.error("No data to export - pipeline failed")
            return False
        
        output_files = {}
        
        # Export CSV
        csv_path = self.paths['output'] / self.config['output_files']['csv']
        if self.io_manager.save_csv(merged_df, csv_path):
            output_files['csv'] = str(csv_path)
        
        # Export Excel with summary
        excel_path = self.paths['output'] / self.config['output_files']['excel']
        if self.io_manager.save_excel(merged_df, excel_path, include_summary=True):
            output_files['excel'] = str(excel_path)
        
        # Export PDF (using external function for complex formatting)
        pdf_path = self.paths['output'] / self.config['output_files']['pdf']
        if self._export_pdf(merged_df, pdf_path):
            output_files['pdf'] = str(pdf_path)
        
        self.metadata['output_files'] = output_files
        self.logger.info(f"Exported data to {len(output_files)} formats")
        
        return len(output_files) > 0
    
    def _export_pdf(self, df, file_path):
        """Export DataFrame to PDF (simplified version)"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            
            doc = SimpleDocTemplate(str(file_path), pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph("Social Media Comments Analysis Report", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))
            
            # Summary
            summary_text = f"""
            Total Comments: {len(df)}<br/>
            Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            Pipeline Run ID: {self.run_id}
            """
            summary = Paragraph(summary_text, styles['Normal'])
            elements.append(summary)
            elements.append(Spacer(1, 20))
            
            # Data table (top 50 rows)
            display_df = df.head(self.config.get('pipeline', {}).get('max_pdf_rows', 50)).copy()
            
            # Truncate long text
            for col in ['text', 'reply']:
                if col in display_df.columns:
                    max_len = self.config.get('pipeline', {}).get('max_text_length', 50)
                    display_df[col] = display_df[col].astype(str).apply(
                        lambda x: x[:max_len] + '...' if len(str(x)) > max_len else x
                    )
            
            # Create table
            table_data = [display_df.columns.tolist()] + display_df.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            doc.build(elements)
            
            self.logger.info(f"PDF exported successfully: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export PDF: {e}")
            return False
    
    def save_metadata(self):
        """Save pipeline execution metadata"""
        metadata_path = self.paths['output'] / self.config['output_files']['metadata']
        
        # Ensure start_time is set
        if self.start_time is None:
            self.start_time = datetime.now()
        
        end_time = datetime.now()
        
        pipeline_metadata = PipelineMetadata(
            pipeline_run_id=self.run_id,
            start_time=self.start_time,
            end_time=end_time,
            processing_time_seconds=(end_time - self.start_time).total_seconds(),
            total_comments=self.metadata.get('transformation', {}).get('initial_count', 0),
            successful_merges=self.metadata.get('transformation', {}).get('final_count', 0),
            preservation_rate=self.metadata.get('transformation', {}).get('preservation_rate', 0.0),
            input_files={
                'comments': str(self.paths['raw'] / 'social_media_files'),
                'classified': str(self.paths['processed'] / self.config['input_files']['classified']),
                'replies': str(self.paths['processed'] / self.config['input_files']['replies'])
            },
            output_files=self.metadata.get('output_files', {}) if self.metadata else {},
            gemini_analyzed_count=self.metadata.get('gemini_analysis', {}).get('success_count'),
            gemini_success_rate=self.metadata.get('gemini_analysis', {}).get('success_rate')
        )
        
        self.io_manager.save_json(pipeline_metadata.model_dump(), metadata_path)
        self.logger.info(f"Pipeline metadata saved to {metadata_path}")
    
    def run_pipeline(self, create_sample: Optional[bool] = None):
        """Run the complete ETL pipeline"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting ETL Pipeline run {self.run_id}...")
        
        try:
            # Determine if we should create sample data
            if create_sample is None:
                create_sample = self.config.get('pipeline', {}).get('create_sample_data', False)
            
            # Skip sample data creation since we're using real files
            if create_sample:
                self.logger.info("Sample data creation skipped - using real social media files")
            
            # ETL phases
            df_comments, df_classified, df_replies = self.extract_data()
            
            if df_comments.empty:
                self.logger.error("No comments data extracted - pipeline cannot continue")
                return False
                
            merged_df = self.transform_data(df_comments, df_classified, df_replies)
            export_success = self.load_data(merged_df)
            
            # Save metadata
            self.save_metadata()
            
            # Calculate final processing time
            processing_time = (datetime.now() - self.start_time).total_seconds()
            
            if export_success:
                self.logger.info(f"✅ ETL Pipeline completed successfully in {processing_time:.2f} seconds")
                self.logger.info(f"📊 Processed {self.metadata.get('transformation', {}).get('final_count', 0)} comments")
                if 'gemini_analysis' in self.metadata:
                    gemini_stats = self.metadata['gemini_analysis']
                    self.logger.info(f"🤖 Gemini analysis: {gemini_stats['success_count']}/{gemini_stats['total_processed']} successful ({gemini_stats['success_rate']:.1%})")
                return True
            else:
                self.logger.error("❌ Pipeline completed with export failures")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {str(e)}")
            return False

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Social Media Comments ETL Pipeline with Gemini AI')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--input-path', help='Override input data path')
    parser.add_argument('--output-path', help='Override output path')
    parser.add_argument('--disable-gemini', action='store_true', help='Disable Gemini AI analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = ETLPipeline(args.config)
        
        # Override Gemini setting if requested
        if args.disable_gemini:
            pipeline.enable_gemini = False
            pipeline.logger.info("Gemini AI analysis disabled via command line")
        
        # Override paths if specified
        if args.input_path:
            pipeline.paths['raw'] = Path(args.input_path)
        if args.output_path:
            pipeline.paths['output'] = Path(args.output_path)
        
        # Run pipeline
        success = pipeline.run_pipeline(create_sample=False)
        
        if success:
            print("✅ ETL Pipeline completed successfully!")
            print(f"📁 Processed files: {', '.join(pipeline.metadata.get('source_files', []))}")
            print("📊 Generated outputs:")
            for format_name, file_path in pipeline.metadata.get('output_files', {}).items():
                print(f"   - {format_name.upper()}: {file_path}")
            
            # Show Gemini analysis results
            if 'gemini_analysis' in pipeline.metadata:
                gemini_stats = pipeline.metadata['gemini_analysis']
                print(f"🤖 Gemini Analysis Results:")
                print(f"   - Processed: {gemini_stats['total_processed']} comments")
                print(f"   - Success rate: {gemini_stats['success_rate']:.1%}")
                print(f"   - Processing time: {gemini_stats['processing_time']:.1f}s")
        else:
            print("❌ ETL Pipeline failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()