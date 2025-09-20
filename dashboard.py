#!/usr/bin/env python3
"""
Social Media Comments Analysis Dashboard
Author: Yerassyl
Description: Interactive dashboard for visualizing processed comments data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Social Media Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Dashboard:
    def __init__(self):
        self.data = None
        self.base_path = Path("./")
        self.output_path = self.base_path / "output"
    
    def load_data(self, file_path=None):
        """Load data from CSV file or use default path"""
        try:
            if file_path is None:
                file_path = self.output_path / "final.csv"
            
            if Path(file_path).exists():
                self.data = pd.read_csv(file_path, encoding='utf-8')
                # Convert timestamp if present
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], errors='coerce')
                return True
            else:
                st.error(f"Data file not found: {file_path}")
                return False
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def create_sample_data(self):
        """Create sample data for demo mode"""
        sample_data = {
            'id': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'],
            'author': ['user123', 'user456', 'user789', 'user101', 'user202', 'user303', 'user404', 'user505', 'user606', 'user707'],
            'timestamp': pd.date_range('2025-09-20 10:00:00', periods=10, freq='5min'),
            'text': [
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
            'lang': ['en', 'en', 'kk', 'ru', 'mixed', 'en', 'mixed', 'en', 'en', 'en'],
            'moderation': ['safe', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe'],
            'category': ['complaint', 'thanks', 'review', 'review', 'thanks', 'complaint', 'thanks', 'question', 'complaint', 'thanks'],
            'sentiment': ['negative', 'positive', 'positive', 'neutral', 'positive', 'negative', 'positive', 'neutral', 'negative', 'positive'],
            'reply': [
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
        self.data = pd.DataFrame(sample_data)
        return True
    
    def render_header(self):
        """Render dashboard header"""
        st.title("📊 Social Media Comments Analysis Dashboard")
        st.markdown("---")
        
        # Data loading section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Data Source")
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col3:
            demo_mode = st.checkbox("Demo Mode", value=True)
        
        return demo_mode
    
    def render_metrics(self):
        """Render key metrics cards"""
        if self.data is None or self.data.empty:
            st.warning("No data available")
            return
        
        # Calculate metrics
        total_comments = len(self.data)
        languages = self.data['lang'].nunique() if 'lang' in self.data.columns else 0
        replied_comments = self.data['reply'].notna().sum() if 'reply' in self.data.columns else 0
        positive_sentiment = (self.data['sentiment'] == 'positive').sum() if 'sentiment' in self.data.columns else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", total_comments)
        
        with col2:
            st.metric("Languages Detected", languages)
        
        with col3:
            reply_rate = (replied_comments / total_comments * 100) if total_comments > 0 else 0
            st.metric("Reply Rate", f"{reply_rate:.1f}%")
        
        with col4:
            positive_rate = (positive_sentiment / total_comments * 100) if total_comments > 0 else 0
            st.metric("Positive Sentiment", f"{positive_rate:.1f}%")
    
    def render_charts(self):
        """Render visualization charts"""
        if self.data is None or self.data.empty:
            st.warning("No data available for visualization")
            return
        
        col1, col2 = st.columns(2)
        
        # Sentiment Distribution Pie Chart
        with col1:
            st.subheader("📈 Sentiment Distribution")
            if 'sentiment' in self.data.columns:
                sentiment_counts = self.data['sentiment'].value_counts()
                
                # Define colors for sentiments
                colors = {'positive': '#00CC96', 'negative': '#EF553B', 'neutral': '#636EFA', 'unknown': '#AB63FA'}
                color_sequence = [colors.get(sentiment, '#FFA15A') for sentiment in sentiment_counts.index]
                
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Comment Sentiments",
                    color_discrete_sequence=color_sequence
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, width='stretch')
            else:
                st.info("Sentiment data not available")
        
        # Category Distribution Bar Chart
        with col2:
            st.subheader("📊 Category Distribution")
            if 'category' in self.data.columns:
                category_counts = self.data['category'].value_counts()
                
                fig_bar = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Comment Categories",
                    labels={'x': 'Category', 'y': 'Count'},
                    color=category_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.info("Category data not available")
        
        # Language Distribution
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🌍 Language Distribution")
            if 'lang' in self.data.columns:
                lang_counts = self.data['lang'].value_counts()
                
                fig_lang = px.bar(
                    x=lang_counts.values,
                    y=lang_counts.index,
                    orientation='h',
                    title="Languages Detected",
                    labels={'x': 'Count', 'y': 'Language'},
                    color=lang_counts.values,
                    color_continuous_scale='Blues'
                )
                fig_lang.update_layout(showlegend=False)
                st.plotly_chart(fig_lang, width='stretch')
            else:
                st.info("Language data not available")
        
        # Timeline Chart (if timestamp available)
        with col4:
            st.subheader("📅 Comments Timeline")
            if 'timestamp' in self.data.columns and pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
                # Group by hour for timeline
                self.data['hour'] = self.data['timestamp'].dt.floor('h')
                timeline_data = self.data.groupby('hour').size().reset_index(name='count')
                
                fig_timeline = px.line(
                    timeline_data,
                    x='hour',
                    y='count',
                    title="Comments Over Time",
                    labels={'hour': 'Time', 'count': 'Number of Comments'},
                    markers=True
                )
                st.plotly_chart(fig_timeline, width='stretch')
            else:
                st.info("Timeline data not available")
    
    def render_data_table(self):
        """Render interactive data table"""
        if self.data is None or self.data.empty:
            st.warning("No data available")
            return
        
        st.subheader("📋 Comments & Replies Data")
        
        # Filters in sidebar
        st.sidebar.subheader("🔍 Filters")
        
        # Sentiment filter
        if 'sentiment' in self.data.columns:
            selected_sentiments = st.sidebar.multiselect(
                "Filter by Sentiment",
                options=self.data['sentiment'].unique(),
                default=self.data['sentiment'].unique()
            )
        else:
            selected_sentiments = []
        
        # Category filter
        if 'category' in self.data.columns:
            selected_categories = st.sidebar.multiselect(
                "Filter by Category",
                options=self.data['category'].unique(),
                default=self.data['category'].unique()
            )
        else:
            selected_categories = []
        
        # Language filter
        if 'lang' in self.data.columns:
            selected_languages = st.sidebar.multiselect(
                "Filter by Language",
                options=self.data['lang'].unique(),
                default=self.data['lang'].unique()
            )
        else:
            selected_languages = []
        
        # Apply filters
        filtered_data = self.data.copy()
        
        if selected_sentiments and 'sentiment' in self.data.columns:
            filtered_data = filtered_data[filtered_data['sentiment'].isin(selected_sentiments)]
        
        if selected_categories and 'category' in self.data.columns:
            filtered_data = filtered_data[filtered_data['category'].isin(selected_categories)]
        
        if selected_languages and 'lang' in self.data.columns:
            filtered_data = filtered_data[filtered_data['lang'].isin(selected_languages)]
        
        # Display filtered data
        st.write(f"Showing {len(filtered_data)} of {len(self.data)} comments")
        
        # Format display columns
        display_columns = ['id', 'author', 'text', 'sentiment', 'category', 'reply']
        available_columns = [col for col in display_columns if col in filtered_data.columns]
        
        if available_columns:
            display_data = filtered_data[available_columns].copy()
            
            # Truncate long text for better display
            for col in ['text', 'reply']:
                if col in display_data.columns:
                    display_data[col] = display_data[col].astype(str).apply(
                        lambda x: x[:100] + '...' if len(str(x)) > 100 else x
                    )
            
            st.dataframe(
                display_data,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No data columns available for display")
    
    def render_sidebar_info(self):
        """Render sidebar information"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dashboard Info")
        
        if self.data is not None:
            st.sidebar.write(f"**Dataset Shape:** {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            
            if 'timestamp' in self.data.columns:
                try:
                    min_time = self.data['timestamp'].min()
                    max_time = self.data['timestamp'].max()
                    st.sidebar.write(f"**Time Range:** {min_time.strftime('%Y-%m-%d %H:%M')} to {max_time.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.sidebar.write("**Time Range:** Not available")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Actions")
        
        if st.sidebar.button("💾 Download CSV"):
            if self.data is not None:
                csv = self.data.to_csv(index=False)
                st.sidebar.download_button(
                    label="📥 Download Data",
                    data=csv,
                    file_name=f"comments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Last Updated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def main():
    """Main dashboard function"""
    dashboard = Dashboard()
    
    # Render header
    demo_mode = dashboard.render_header()
    
    # Load data
    if demo_mode:
        dashboard.create_sample_data()
        st.success("✅ Demo data loaded successfully!")
    else:
        if not dashboard.load_data():
            st.error("❌ Failed to load data. Switching to demo mode.")
            dashboard.create_sample_data()
    
    # Render dashboard components
    if dashboard.data is not None and not dashboard.data.empty:
        dashboard.render_metrics()
        st.markdown("---")
        dashboard.render_charts()
        st.markdown("---")
        dashboard.render_data_table()
        dashboard.render_sidebar_info()
    else:
        st.error("No data available to display")

if __name__ == "__main__":
    main()