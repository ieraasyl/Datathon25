#!/usr/bin/env python3
"""
ETL Pipeline for Social Media Comments Analysis
Author: Yerassyl
Description: Merges data from parser, classifier, and response generator into unified dataset
"""

import json
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, base_path="./"):
        """Initialize ETL Pipeline with base directory path"""
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.output_path = self.base_path / "output"
        
        # Create directories if they don't exist
        self.data_path.mkdir(exist_ok=True)
        (self.data_path / "raw").mkdir(exist_ok=True)
        (self.data_path / "processed").mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        
        logger.info(f"ETL Pipeline initialized with base path: {self.base_path}")

    def load_json_data(self, file_path):
        """Load JSON data from file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. Using empty list.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def create_sample_data(self):
        """Create sample data files for testing (placeholder data)"""
        logger.info("Creating sample data files...")
        
        # Sample comments (from Daulet's parser)
        sample_comments = [
            {"id": "c1", "author": "user123", "timestamp": "2025-09-20T10:00:00", "text": "Bad network! Connection keeps dropping."},
            {"id": "c2", "author": "user456", "timestamp": "2025-09-20T10:05:00", "text": "Thanks for the fast internet!"},
            {"id": "c3", "author": "user789", "timestamp": "2025-09-20T10:10:00", "text": "Актубе филиалынын интернеті жақсы"},
            {"id": "c4", "author": "user101", "timestamp": "2025-09-20T10:15:00", "text": "Тарифы дорогие, но качество хорошее"},
            {"id": "c5", "author": "user202", "timestamp": "2025-09-20T10:20:00", "text": "Great customer service! Молодцы!"}
        ]
        
        # Sample classifications (from Olzhas)
        sample_classifications = [
            {"id": "c1", "lang": "en", "moderation": "safe", "category": "complaint", "sentiment": "negative"},
            {"id": "c2", "lang": "en", "moderation": "safe", "category": "thanks", "sentiment": "positive"},
            {"id": "c3", "lang": "kk", "moderation": "safe", "category": "review", "sentiment": "positive"},
            {"id": "c4", "lang": "ru", "moderation": "safe", "category": "review", "sentiment": "neutral"},
            {"id": "c5", "lang": "mixed", "moderation": "safe", "category": "thanks", "sentiment": "positive"}
        ]
        
        # Sample replies (from Anuar)
        sample_replies = [
            {"id": "c1", "reply": "Кешіріңіз! We're working to improve network stability. Please contact 109 for support."},
            {"id": "c2", "reply": "Рахмет за отзыв! We're glad you like our service 😊"},
            {"id": "c3", "reply": "Рахмет! Біз Ақтөбеде қызметті жақсартуға тырысамыз!"},
            {"id": "c4", "reply": "Thank you for feedback! We offer various tariff options to suit different needs."},
            {"id": "c5", "reply": "Спасибо! Our team works hard to provide excellent service 🙏"}
        ]
        
        # Save sample files
        with open(self.data_path / "raw" / "comments.json", 'w', encoding='utf-8') as f:
            json.dump(sample_comments, f, ensure_ascii=False, indent=2)
        
        with open(self.data_path / "processed" / "classified.json", 'w', encoding='utf-8') as f:
            json.dump(sample_classifications, f, ensure_ascii=False, indent=2)
            
        with open(self.data_path / "processed" / "replies.json", 'w', encoding='utf-8') as f:
            json.dump(sample_replies, f, ensure_ascii=False, indent=2)
        
        logger.info("Sample data files created successfully")

    def extract_data(self):
        """Extract data from all three sources"""
        logger.info("Starting data extraction...")
        
        # Load data from each source
        comments = self.load_json_data(self.data_path / "raw" / "comments.json")
        classifications = self.load_json_data(self.data_path / "processed" / "classified.json")
        replies = self.load_json_data(self.data_path / "processed" / "replies.json")
        
        # Convert to DataFrames for easier merging
        df_comments = pd.DataFrame(comments) if comments else pd.DataFrame()
        df_classifications = pd.DataFrame(classifications) if classifications else pd.DataFrame()
        df_replies = pd.DataFrame(replies) if replies else pd.DataFrame()
        
        logger.info(f"Extracted: {len(df_comments)} comments, {len(df_classifications)} classifications, {len(df_replies)} replies")
        
        return df_comments, df_classifications, df_replies

    def transform_data(self, df_comments, df_classifications, df_replies):
        """Transform and merge data from all sources"""
        logger.info("Starting data transformation...")
        
        if df_comments.empty:
            logger.warning("No comments data found. Cannot proceed with transformation.")
            return pd.DataFrame()
        
        # Start with comments as base
        merged_df = df_comments.copy()
        
        # Merge with classifications
        if not df_classifications.empty:
            merged_df = merged_df.merge(df_classifications, on='id', how='left')
            logger.info("Merged classifications data")
        else:
            # Add empty classification columns
            merged_df['lang'] = None
            merged_df['moderation'] = None
            merged_df['category'] = None
            merged_df['sentiment'] = None
            logger.warning("No classifications data - added empty columns")
        
        # Merge with replies
        if not df_replies.empty:
            merged_df = merged_df.merge(df_replies, on='id', how='left')
            logger.info("Merged replies data")
        else:
            # Add empty reply column
            merged_df['reply'] = None
            logger.warning("No replies data - added empty column")
        
        # Data quality checks and transformations
        initial_count = len(merged_df)
        
        # Convert timestamp to datetime
        if 'timestamp' in merged_df.columns:
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], errors='coerce')
        
        # Fill NaN values
        fill_values = {
            'lang': 'unknown',
            'moderation': 'unknown',
            'category': 'unknown',
            'sentiment': 'unknown',
            'reply': ''
        }
        
        for col, fill_val in fill_values.items():
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(fill_val)
        
        # Remove duplicates
        merged_df = merged_df.drop_duplicates(subset=['id'])
        final_count = len(merged_df)
        
        preservation_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
        logger.info(f"Data transformation completed. Preservation rate: {preservation_rate:.1f}%")
        
        return merged_df

    def export_csv(self, df, filename="final.csv"):
        """Export DataFrame to CSV"""
        output_file = self.output_path / filename
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"CSV exported: {output_file}")
        return output_file

    def export_excel(self, df, filename="final.xlsx"):
        """Export DataFrame to Excel"""
        output_file = self.output_path / filename
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Comments_Analysis', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Comments', 'Languages', 'Sentiments', 'Categories', 'Replied Comments'],
                'Count': [
                    len(df),
                    df['lang'].nunique() if 'lang' in df.columns else 0,
                    df['sentiment'].nunique() if 'sentiment' in df.columns else 0,
                    df['category'].nunique() if 'category' in df.columns else 0,
                    df['reply'].notna().sum() if 'reply' in df.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel exported: {output_file}")
        return output_file

    def export_pdf(self, df, filename="final.pdf"):
        """Export DataFrame to PDF report"""
        output_file = self.output_path / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_file), pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        title = Paragraph("Social Media Comments Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Summary section
        summary_style = styles['Heading2']
        summary_title = Paragraph("Summary", summary_style)
        elements.append(summary_title)
        
        # Summary statistics
        total_comments = len(df)
        languages = df['lang'].nunique() if 'lang' in df.columns else 0
        replied = df['reply'].notna().sum() if 'reply' in df.columns else 0
        
        summary_text = f"""
        Total Comments: {total_comments}<br/>
        Languages Detected: {languages}<br/>
        Comments with Replies: {replied}<br/>
        Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        summary_para = Paragraph(summary_text, styles['Normal'])
        elements.append(summary_para)
        elements.append(Spacer(1, 20))
        
        # Data table (first 50 rows to fit in PDF)
        table_title = Paragraph("Comments Data (Top 50 entries)", summary_style)
        elements.append(table_title)
        
        # Prepare table data
        display_df = df.head(50).copy()
        
        # Truncate long text for PDF display
        for col in ['text', 'reply']:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str).apply(
                    lambda x: x[:50] + '...' if len(str(x)) > 50 else x
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
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        logger.info(f"PDF exported: {output_file}")
        return output_file

    def run_pipeline(self, create_sample=True):
        """Run the complete ETL pipeline"""
        start_time = datetime.now()
        logger.info("Starting ETL Pipeline...")
        
        try:
            # Create sample data if requested
            if create_sample:
                self.create_sample_data()
            
            # Extract
            df_comments, df_classifications, df_replies = self.extract_data()
            
            # Transform
            merged_df = self.transform_data(df_comments, df_classifications, df_replies)
            
            if merged_df.empty:
                logger.error("No data to export. Pipeline failed.")
                return False
            
            # Load (Export)
            csv_file = self.export_csv(merged_df)
            excel_file = self.export_excel(merged_df)
            pdf_file = self.export_pdf(merged_df)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"ETL Pipeline completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Exported files: {csv_file.name}, {excel_file.name}, {pdf_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            return False

def main():
    """Main function to run the ETL pipeline"""
    pipeline = ETLPipeline()
    
    # Run with sample data creation
    success = pipeline.run_pipeline(create_sample=True)
    
    if success:
        print("✅ ETL Pipeline completed successfully!")
        print("📁 Check the 'output' folder for generated reports:")
        print("   - final.csv (UTF-8 encoded)")
        print("   - final.xlsx (Excel format with summary)")
        print("   - final.pdf (Formatted report)")
    else:
        print("❌ ETL Pipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()