#!/usr/bin/env python3
"""
Social Media Comments Analysis Dashboard
Author: Yerassyl
Description: Streamlit dashboard with caching and download capabilities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import Optional
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from constants import *
from utils.io import load_config
from utils.logging import setup_logging, get_logger

# Setup logging and configuration
setup_logging()
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=UI_CONSTANTS.get('title', "Social Media Analysis Dashboard"),
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Dashboard:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.data = None
        self.metadata = None
        self.base_path = Path(self.config['data']['base_path'])
        self.output_path = self.base_path / self.config['data']['output_path']
        logger.info("Dashboard initialized")
    
    @st.cache_data(ttl=3600, show_spinner=True)
    def load_data(_self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load data with caching for performance"""
        try:
            if file_path is None:
                file_path = _self.output_path / _self.config['output_files']['csv']
            
            if Path(file_path).exists():
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # Data preprocessing
                if COLUMNS['TIMESTAMP'] in df.columns:
                    df[COLUMNS['TIMESTAMP']] = pd.to_datetime(df[COLUMNS['TIMESTAMP']], errors='coerce')
                
                logger.info(f"Data loaded successfully: {len(df)} records")
                return df
            else:
                logger.warning(f"Data file not found: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_metadata(_self) -> dict:
        """Load pipeline metadata"""
        try:
            metadata_path = _self.output_path / _self.config['output_files']['metadata']
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demo mode"""
        sample_data = {
            COLUMNS['ID']: ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'],
            COLUMNS['AUTHOR']: ['user123', 'user456', 'user789', 'user101', 'user202', 
                               'user303', 'user404', 'user505', 'user606', 'user707'],
            COLUMNS['TIMESTAMP']: pd.date_range('2025-09-20 10:00:00', periods=10, freq='5min'),
            COLUMNS['TEXT']: [
                'Bad network! Connection keeps dropping.',
                'Thanks for the fast internet!',
                'Актубе филиалынын интернеті жақсы',
                'Тарифы дорогие, но качество хорошее',
                'Great customer service! Молодцы!',
                'Internet speed is very slow in Almaty',
                'Рахмет за хорошее покрытие в Астане!',
                'When will 5G be available?',
                'Billing system has bugs',
                'Perfect service, keep it up!'
            ],
            COLUMNS['LANGUAGE']: ['en', 'en', 'kk', 'ru', 'mixed', 'en', 'mixed', 'en', 'en', 'en'],
            COLUMNS['MODERATION']: ['safe'] * 10,
            COLUMNS['CATEGORY']: ['complaint', 'thanks', 'review', 'review', 'thanks', 
                                 'complaint', 'thanks', 'question', 'complaint', 'thanks'],
            COLUMNS['SENTIMENT']: ['negative', 'positive', 'positive', 'neutral', 'positive', 
                                  'negative', 'positive', 'neutral', 'negative', 'positive'],
            COLUMNS['REPLY']: [
                'We apologize for the inconvenience. Please contact support.',
                'Thank you for your feedback! 😊',
                'Рахмет! Біз қызметті жақсартуға тырысамыз!',
                'We offer various tariff options to suit your needs.',
                'Thank you! Our team works hard to provide excellent service.',
                'We are working to improve network coverage in Almaty.',
                'Рахмет за отзыв! Мы стараемся лучше!',
                '5G rollout is planned for 2025. Stay tuned!',
                'Thank you for reporting. Our team will investigate.',
                'We appreciate your support! 🙏'
            ]
        }
        df = pd.DataFrame(sample_data)
        logger.info("Sample data created for demo mode")
        return df
    
    def render_header(self):
        """Render dashboard header with data source controls"""
        st.title("📊 Social Media Comments Analysis Dashboard")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.subheader("Data Source")
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        with col3:
            demo_mode = st.checkbox("Demo Mode", value=self.config['dashboard'].get('demo_mode', True))
        
        with col4:
            show_metadata = st.checkbox("Show Metadata", value=False)
        
        return demo_mode, show_metadata
    
    def render_metadata_info(self, metadata: dict):
        """Render pipeline metadata information"""
        if not metadata:
            st.info("No pipeline metadata available")
            return
        
        st.subheader("🔧 Pipeline Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pipeline Run ID", metadata.get('pipeline_run_id', 'N/A'))
            processing_time = metadata.get('processing_time_seconds', 0)
            st.metric("Processing Time", f"{processing_time:.2f}s")
        
        with col2:
            preservation_rate = metadata.get('preservation_rate', 0)
            st.metric("Data Preservation Rate", f"{preservation_rate:.1f}%")
            total_comments = metadata.get('total_comments', 0)
            st.metric("Total Input Comments", total_comments)
        
        with col3:
            start_time = metadata.get('start_time', '')
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', ''))
                    st.metric("Pipeline Started", start_dt.strftime('%Y-%m-%d %H:%M'))
                except:
                    st.metric("Pipeline Started", "N/A")
        
        st.markdown("---")
    
    def render_metrics(self, df: pd.DataFrame, metadata: dict = None):
        """Render key performance metrics"""
        if df.empty:
            st.warning("No data available for metrics")
            return
        
        # Calculate metrics
        total_comments = len(df)
        languages = df[COLUMNS['LANGUAGE']].nunique() if COLUMNS['LANGUAGE'] in df.columns else 0
        replied_comments = df[COLUMNS['REPLY']].notna().sum() if COLUMNS['REPLY'] in df.columns else 0
        positive_sentiment = (df[COLUMNS['SENTIMENT']] == 'positive').sum() if COLUMNS['SENTIMENT'] in df.columns else 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(METRIC_LABELS['TOTAL_COMMENTS'], total_comments)
        
        with col2:
            st.metric(METRIC_LABELS['LANGUAGES_DETECTED'], languages)
        
        with col3:
            reply_rate = (replied_comments / total_comments * 100) if total_comments > 0 else 0
            st.metric(METRIC_LABELS['REPLY_RATE'], f"{reply_rate:.1f}%")
        
        with col4:
            positive_rate = (positive_sentiment / total_comments * 100) if total_comments > 0 else 0
            st.metric(METRIC_LABELS['POSITIVE_SENTIMENT'], f"{positive_rate:.1f}%")
        
        # Additional metrics from metadata
        if metadata:
            col5, col6 = st.columns(2)
            with col5:
                processing_time = metadata.get('processing_time_seconds', 0)
                st.metric(METRIC_LABELS['PROCESSING_TIME'], f"{processing_time:.2f}s")
            with col6:
                preservation_rate = metadata.get('preservation_rate', 0)
                st.metric(METRIC_LABELS['PRESERVATION_RATE'], f"{preservation_rate:.1f}%")
    
    def render_charts(self, df: pd.DataFrame):
        """Render interactive visualization charts"""
        if df.empty:
            st.warning("No data available for visualization")
            return
        
        # First row: Sentiment and Category
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Sentiment Distribution")
            if COLUMNS['SENTIMENT'] in df.columns:
                sentiment_counts = df[COLUMNS['SENTIMENT']].value_counts()
                
                colors = [SENTIMENTS.get(sentiment, '#FFA15A') for sentiment in sentiment_counts.index]
                
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Comment Sentiments",
                    color_discrete_sequence=colors,
                    height=UI_CONSTANTS['CHART_HEIGHT']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Sentiment data not available")
        
        with col2:
            st.subheader("📊 Category Distribution")
            if COLUMNS['CATEGORY'] in df.columns:
                category_counts = df[COLUMNS['CATEGORY']].value_counts()
                
                fig_bar = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Comment Categories",
                    labels={'x': 'Category', 'y': 'Count'},
                    color=category_counts.values,
                    color_continuous_scale='Viridis',
                    height=UI_CONSTANTS['CHART_HEIGHT']
                )
                fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Category data not available")
        
        # Second row: Language and Timeline
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🌍 Language Distribution")
            if COLUMNS['LANGUAGE'] in df.columns:
                lang_counts = df[COLUMNS['LANGUAGE']].value_counts()
                lang_display = [LANGUAGE_MAPPING.get(lang, lang.title()) for lang in lang_counts.index]
                
                fig_lang = px.bar(
                    x=lang_counts.values,
                    y=lang_display,
                    orientation='h',
                    title="Languages Detected",
                    labels={'x': 'Count', 'y': 'Language'},
                    color=lang_counts.values,
                    color_continuous_scale='Blues',
                    height=UI_CONSTANTS['CHART_HEIGHT']
                )
                fig_lang.update_layout(showlegend=False)
                st.plotly_chart(fig_lang, use_container_width=True)
            else:
                st.info("Language data not available")
        
        with col4:
            st.subheader("📅 Comments Timeline")
            if COLUMNS['TIMESTAMP'] in df.columns and pd.api.types.is_datetime64_any_dtype(df[COLUMNS['TIMESTAMP']]):
                # Group by hour for timeline
                df_copy = df.copy()
                df_copy['hour'] = df_copy[COLUMNS['TIMESTAMP']].dt.floor('h')
                timeline_data = df_copy.groupby('hour').size().reset_index(name='count')
                
                fig_timeline = px.line(
                    timeline_data,
                    x='hour',
                    y='count',
                    title="Comments Over Time",
                    labels={'hour': 'Time', 'count': 'Number of Comments'},
                    markers=True,
                    height=UI_CONSTANTS['CHART_HEIGHT']
                )
                fig_timeline.update_layout(showlegend=False)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("Timeline data not available")
    
    def render_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render sidebar filters and return filtered dataframe"""
        st.sidebar.subheader("🔍 Filters")
        
        filtered_df = df.copy()
        
        # Sentiment filter
        if COLUMNS['SENTIMENT'] in df.columns:
            sentiments = df[COLUMNS['SENTIMENT']].unique()
            selected_sentiments = st.sidebar.multiselect(
                "Filter by Sentiment",
                options=sentiments,
                default=sentiments
            )
            if selected_sentiments:
                filtered_df = filtered_df[filtered_df[COLUMNS['SENTIMENT']].isin(selected_sentiments)]
        
        # Category filter
        if COLUMNS['CATEGORY'] in df.columns:
            categories = df[COLUMNS['CATEGORY']].unique()
            selected_categories = st.sidebar.multiselect(
                "Filter by Category",
                options=categories,
                default=categories
            )
            if selected_categories:
                filtered_df = filtered_df[filtered_df[COLUMNS['CATEGORY']].isin(selected_categories)]
        
        # Language filter
        if COLUMNS['LANGUAGE'] in df.columns:
            languages = df[COLUMNS['LANGUAGE']].unique()
            lang_display = {lang: LANGUAGE_MAPPING.get(lang, lang.title()) for lang in languages}
            selected_languages = st.sidebar.multiselect(
                "Filter by Language",
                options=languages,
                default=languages,
                format_func=lambda x: lang_display.get(x, str(x))
            )
            if selected_languages:
                filtered_df = filtered_df[filtered_df[COLUMNS['LANGUAGE']].isin(selected_languages)]
        
        # Text search
        search_term = st.sidebar.text_input("Search in Comments", "")
        if search_term and COLUMNS['TEXT'] in filtered_df.columns:
            mask = filtered_df[COLUMNS['TEXT']].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def render_data_table(self, df: pd.DataFrame):
        """Render interactive data table"""
        if df.empty:
            st.warning("No data to display")
            return
        
        st.subheader("📋 Comments & Replies Data")
        
        # Show record count
        st.write(f"Showing **{len(df)}** comments")
        
        # Select display columns
        display_columns = [col for col in [
            COLUMNS['ID'], COLUMNS['AUTHOR'], COLUMNS['TEXT'], 
            COLUMNS['SENTIMENT'], COLUMNS['CATEGORY'], COLUMNS['LANGUAGE'],
            COLUMNS['REPLY']
        ] if col in df.columns]
        
        if not display_columns:
            st.error("No valid columns found for display")
            return
        
        # Format data for display
        display_df = df[display_columns].copy()
        
        # Truncate long text fields for better display
        text_columns = [COLUMNS['TEXT'], COLUMNS['REPLY']]
        for col in text_columns:
            if col in display_df.columns:
                max_len = UI_CONSTANTS['MAX_TEXT_DISPLAY']
                display_df[col] = display_df[col].astype(str).apply(
                    lambda x: x[:max_len] + '...' if len(str(x)) > max_len else x
                )
        
        # Display with pagination
        st.dataframe(
            display_df,
            use_container_width=True,
            height=UI_CONSTANTS['TABLE_HEIGHT']
        )
    
    def render_download_buttons(self, df: pd.DataFrame):
        """Render download buttons for different formats"""
        if df.empty:
            st.info("No data available for download")
            return
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Download Data")
        
        # CSV download
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"comments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Excel download
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Comments_Data', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Comments', 'Languages', 'Sentiments', 'Categories'],
                'Count': [
                    len(df),
                    df[COLUMNS['LANGUAGE']].nunique() if COLUMNS['LANGUAGE'] in df.columns else 0,
                    df[COLUMNS['SENTIMENT']].nunique() if COLUMNS['SENTIMENT'] in df.columns else 0,
                    df[COLUMNS['CATEGORY']].nunique() if COLUMNS['CATEGORY'] in df.columns else 0
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_buffer.seek(0)
        
        st.sidebar.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"comments_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Show existing files if available
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Generated Files")
        
        output_files = ['final.csv', 'final.xlsx', 'final.pdf']
        for filename in output_files:
            file_path = self.output_path / filename
            if file_path.exists():
                st.sidebar.text(f"✅ {filename}")
            else:
                st.sidebar.text(f"❌ {filename}")
    
    def render_sidebar_info(self, df: pd.DataFrame):
        """Render additional sidebar information"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dataset Info")
        
        if not df.empty:
            st.sidebar.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.sidebar.write(f"**Memory:** {memory_mb:.2f} MB")
            
            # Time range
            if COLUMNS['TIMESTAMP'] in df.columns and pd.api.types.is_datetime64_any_dtype(df[COLUMNS['TIMESTAMP']]):
                try:
                    min_time = df[COLUMNS['TIMESTAMP']].min()
                    max_time = df[COLUMNS['TIMESTAMP']].max()
                    st.sidebar.write(f"**Time Range:**")
                    st.sidebar.write(f"From: {min_time.strftime('%Y-%m-%d %H:%M')}")
                    st.sidebar.write(f"To: {max_time.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.sidebar.write("**Time Range:** Not available")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main dashboard application"""
    try:
        dashboard = Dashboard()
        
        # Render header and get mode settings
        demo_mode, show_metadata = dashboard.render_header()
        
        # Load data based on mode
        if demo_mode:
            df = dashboard.create_sample_data()
            metadata = {}
            st.success("✅ Demo data loaded successfully!")
        else:
            df = dashboard.load_data()
            metadata = dashboard.load_metadata()
            
            if df.empty:
                st.error("❌ No data found. Switching to demo mode...")
                df = dashboard.create_sample_data()
                metadata = {}
        
        # Show metadata if requested
        if show_metadata and metadata:
            dashboard.render_metadata_info(metadata)
        
        # Main dashboard content
        if not df.empty:
            # Apply filters
            filtered_df = dashboard.render_filters(df)
            
            # Render components
            dashboard.render_metrics(filtered_df, metadata)
            st.markdown("---")
            dashboard.render_charts(filtered_df)
            st.markdown("---")
            dashboard.render_data_table(filtered_df)
            
            # Sidebar components
            dashboard.render_download_buttons(filtered_df)
            dashboard.render_sidebar_info(filtered_df)
        
        else:
            st.error("❌ No data available to display")
            
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")

if __name__ == "__main__":
    main()