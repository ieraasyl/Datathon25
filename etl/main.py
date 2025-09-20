#!/usr/bin/env python3
"""
ETL Pipeline Main Module
Author: Yerassyl
Description: Modularized ETL pipeline with CLI interface
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import uuid
import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from constants import *
from models import PipelineMetadata, validate_comments, validate_classified, validate_replies
from utils.io import FileIOManager, load_config
from utils.logging import setup_logging, get_logger, timed_operation

class ETLPipeline:
    """Modularized ETL Pipeline with validation and error handling"""
    
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
        
        self.logger.info(f"ETL Pipeline initialized with run ID: {self.run_id}")
    
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
    
    def create_sample_data(self):
        """Create sample data for testing"""
        self.logger.info("Creating sample data files...")
        
        # Sample comments (Daulet's parser output)
        sample_comments = [
            {"id": "c1", "author": "user123", "timestamp": "2025-09-20T10:00:00", "text": "Bad network! Connection keeps dropping."},
            {"id": "c2", "author": "user456", "timestamp": "2025-09-20T10:05:00", "text": "Thanks for the fast internet!"},
            {"id": "c3", "author": "user789", "timestamp": "2025-09-20T10:10:00", "text": "Актубе филиалынын интернеті жақсы"},
            {"id": "c4", "author": "user101", "timestamp": "2025-09-20T10:15:00", "text": "Тарифы дорогие, но качество хорошее"},
            {"id": "c5", "author": "user202", "timestamp": "2025-09-20T10:20:00", "text": "Great customer service! Молодцы!"},
            {"id": "c6", "author": "user303", "timestamp": "2025-09-20T10:25:00", "text": "Internet speed is very slow in Almaty"},
            {"id": "c7", "author": "user404", "timestamp": "2025-09-20T10:30:00", "text": "Рахмет за хорошее покрытие в Астане!"},
            {"id": "c8", "author": "user505", "timestamp": "2025-09-20T10:35:00", "text": "When will 5G be available?"},
            {"id": "c9", "author": "user606", "timestamp": "2025-09-20T10:40:00", "text": "Billing system has bugs"},
            {"id": "c10", "author": "user707", "timestamp": "2025-09-20T10:45:00", "text": "Perfect service, keep it up!"}
        ]
        
        # Sample classifications (Olzhas's output)
        sample_classifications = [
            {"id": "c1", "lang": "en", "moderation": "safe", "category": "complaint", "sentiment": "negative"},
            {"id": "c2", "lang": "en", "moderation": "safe", "category": "thanks", "sentiment": "positive"},
            {"id": "c3", "lang": "kk", "moderation": "safe", "category": "review", "sentiment": "positive"},
            {"id": "c4", "lang": "ru", "moderation": "safe", "category": "review", "sentiment": "neutral"},
            {"id": "c5", "lang": "mixed", "moderation": "safe", "category": "thanks", "sentiment": "positive"},
            {"id": "c6", "lang": "en", "moderation": "safe", "category": "complaint", "sentiment": "negative"},
            {"id": "c7", "lang": "mixed", "moderation": "safe", "category": "thanks", "sentiment": "positive"},
            {"id": "c8", "lang": "en", "moderation": "safe", "category": "question", "sentiment": "neutral"},
            {"id": "c9", "lang": "en", "moderation": "safe", "category": "complaint", "sentiment": "negative"},
            {"id": "c10", "lang": "en", "moderation": "safe", "category": "thanks", "sentiment": "positive"}
        ]
        
        # Sample replies (Anuar's output)
        sample_replies = [
            {"id": "c1", "reply": "Кешіріңіз! We're working to improve network stability. Please contact 109 for support."},
            {"id": "c2", "reply": "Рахмет за отзыв! We're glad you like our service 😊"},
            {"id": "c3", "reply": "Рахмет! Біз Ақтөбеде қызметті жақсартуға тырысамыз!"},
            {"id": "c4", "reply": "Thank you for feedback! We offer various tariff options to suit different needs."},
            {"id": "c5", "reply": "Спасибо! Our team works hard to provide excellent service 🙏"},
            {"id": "c6", "reply": "We are working to improve network coverage in Almaty. Thank you for your patience."},
            {"id": "c7", "reply": "Рахмет за отзыв! Мы стараемся лучше!"},
            {"id": "c8", "reply": "5G rollout is planned for 2025. Stay tuned for updates!"},
            {"id": "c9", "reply": "Thank you for reporting. Our technical team will investigate this issue."},
            {"id": "c10", "reply": "We appreciate your support! Thank you! 🙏"}
        ]
        
        # Save sample files
        self.io_manager.save_json(sample_comments, self.paths['raw'] / self.config['input_files']['comments'])
        self.io_manager.save_json(sample_classifications, self.paths['processed'] / self.config['input_files']['classified'])
        self.io_manager.save_json(sample_replies, self.paths['processed'] / self.config['input_files']['replies'])
        
        self.logger.info("Sample data files created successfully")
    
    @timed_operation(get_logger(__name__), "Data extraction")
    def extract_data(self):
        """Extract and validate data from all sources"""
        self.logger.info("Starting data extraction phase...")
        
        # Load comments with validation
        comments_path = self.paths['raw'] / self.config['input_files']['comments']
        comments_data, comments_validation = self.io_manager.load_json(
            comments_path, validate_comments
        )
        
        # Load classifications with validation
        classified_path = self.paths['processed'] / self.config['input_files']['classified']
        classified_data, classified_validation = self.io_manager.load_json(
            classified_path, validate_classified
        )
        
        # Load replies with validation
        replies_path = self.paths['processed'] / self.config['input_files']['replies']
        replies_data, replies_validation = self.io_manager.load_json(
            replies_path, validate_replies
        )
        
        # Store validation results in metadata
        self.metadata['validation'] = {
            'comments': comments_validation.model_dump(),
            'classified': classified_validation.model_dump(),
            'replies': replies_validation.model_dump()
        }
        
        # Convert to DataFrames
        df_comments = pd.DataFrame(comments_data) if comments_data else pd.DataFrame()
        df_classified = pd.DataFrame(classified_data) if classified_data else pd.DataFrame()
        df_replies = pd.DataFrame(replies_data) if replies_data else pd.DataFrame()
        
        self.logger.info(f"Extracted: {len(df_comments)} comments, {len(df_classified)} classifications, {len(df_replies)} replies")
        
        return df_comments, df_classified, df_replies
    
    @timed_operation(get_logger(__name__), "Data transformation")
    def transform_data(self, df_comments, df_classified, df_replies):
        """Transform and merge data with quality checks"""
        self.logger.info("Starting data transformation phase...")
        
        if df_comments.empty:
            self.logger.warning("No comments data found - cannot proceed with transformation")
            return pd.DataFrame()
        
        initial_count = len(df_comments)
        
        # Start with comments as base
        merged_df = df_comments.copy()
        
        # Merge classifications
        if not df_classified.empty:
            merged_df = merged_df.merge(df_classified, on=COLUMNS['ID'], how='left')
            self.logger.info(f"Merged {len(df_classified)} classification records")
        else:
            # Add empty classification columns with default values
            for col, default_val in FILL_VALUES.items():
                merged_df[col] = default_val
            self.logger.warning("No classification data - added default values")
        
        # Merge replies
        if not df_replies.empty:
            merged_df = merged_df.merge(df_replies, on=COLUMNS['ID'], how='left')
            self.logger.info(f"Merged {len(df_replies)} reply records")
        else:
            merged_df[COLUMNS['REPLY']] = FILL_VALUES['reply']
            self.logger.warning("No reply data - added empty replies")
        
        # Data quality transformations
        merged_df = self._apply_data_quality_fixes(merged_df)
        
        # Calculate preservation rate
        final_count = len(merged_df)
        preservation_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
        
        # Check preservation rate threshold
        threshold = self.config['pipeline']['preservation_rate_threshold']
        if preservation_rate < threshold:
            self.logger.warning(
                ERROR_MESSAGES['LOW_PRESERVATION_RATE'].format(
                    rate=preservation_rate, threshold=threshold
                )
            )
        
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
        
        # Convert timestamp to datetime
        if COLUMNS['TIMESTAMP'] in df.columns:
            df[COLUMNS['TIMESTAMP']] = pd.to_datetime(df[COLUMNS['TIMESTAMP']], errors='coerce')
        
        # Fill missing values with constants
        for col, fill_val in FILL_VALUES.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        
        # Remove duplicates by ID
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[COLUMNS['ID']], keep='first')
        after_dedup = len(df)
        
        if before_dedup != after_dedup:
            self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        # Standardize text fields
        text_columns = [COLUMNS['TEXT'], COLUMNS['REPLY']]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
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
            display_df = df.head(self.config['pipeline'].get('max_pdf_rows', 50)).copy()
            
            # Truncate long text
            for col in [COLUMNS['TEXT'], COLUMNS['REPLY']]:
                if col in display_df.columns:
                    max_len = self.config['pipeline'].get('max_text_length', 50)
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
            
            self.logger.info(SUCCESS_MESSAGES['FILE_EXPORTED'].format(format='PDF', file_path=file_path))
            return True
            
        except Exception as e:
            self.logger.error(ERROR_MESSAGES['EXPORT_FAILED'].format(format='PDF', error=e))
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
                'comments': str(self.paths['raw'] / self.config['input_files']['comments']),
                'classified': str(self.paths['processed'] / self.config['input_files']['classified']),
                'replies': str(self.paths['processed'] / self.config['input_files']['replies'])
            },
            output_files=self.metadata.get('output_files', {}) if self.metadata else {}
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
                create_sample = self.config['pipeline'].get('create_sample_data', True)
            
            if create_sample:
                self.create_sample_data()
            
            # ETL phases
            df_comments, df_classified, df_replies = self.extract_data()
            merged_df = self.transform_data(df_comments, df_classified, df_replies)
            export_success = self.load_data(merged_df)
            
            # Save metadata
            self.save_metadata()
            
            # Calculate final processing time
            processing_time = (datetime.now() - self.start_time).total_seconds()
            
            if export_success:
                self.logger.info(
                    SUCCESS_MESSAGES['PIPELINE_COMPLETED'].format(time=processing_time)
                )
                return True
            else:
                self.logger.error("Pipeline completed with export failures")
                return False
                
        except Exception as e:
            self.logger.error(ERROR_MESSAGES['PIPELINE_FAILED'].format(error=e))
            return False

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Social Media Comments ETL Pipeline')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--sample', action='store_true', help='Create sample data')
    parser.add_argument('--no-sample', action='store_true', help='Skip sample data creation')
    parser.add_argument('--input-path', help='Override input data path')
    parser.add_argument('--output-path', help='Override output path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Handle conflicting sample arguments
    if args.sample and args.no_sample:
        print("Error: --sample and --no-sample cannot both be specified")
        sys.exit(1)
    
    create_sample = None
    if args.sample:
        create_sample = True
    elif args.no_sample:
        create_sample = False
    
    try:
        # Initialize pipeline
        pipeline = ETLPipeline(args.config)
        
        # Override paths if specified
        if args.input_path:
            pipeline.paths['raw'] = Path(args.input_path)
            pipeline.paths['processed'] = Path(args.input_path)
        if args.output_path:
            pipeline.paths['output'] = Path(args.output_path)
        
        # Run pipeline
        success = pipeline.run_pipeline(create_sample=create_sample)
        
        if success:
            print("✅ ETL Pipeline completed successfully!")
            print("📁 Generated files:")
            for format_name, file_path in pipeline.metadata.get('output_files', {}).items():
                print(f"   - {format_name.upper()}: {file_path}")
        else:
            print("❌ ETL Pipeline failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()