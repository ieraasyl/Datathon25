"""
Advanced filtering and analysis examples
"""

import requests
import pandas as pd
import json
from typing import List, Dict

def analyze_csv_results(csv_file_path: str):
    """Analyze scraped comments from CSV file"""
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    print(f"Analysis of {len(df)} comments")
    print("=" * 40)
    
    # Platform breakdown
    print("Comments by Platform:")
    platform_counts = df['platform'].value_counts()
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count} comments")
    
    print()
    
    # Engagement analysis
    print("Engagement Analysis:")
    avg_likes = df['likes_count'].mean()
    avg_replies = df['replies_count'].mean()
    print(f"  Average likes per comment: {avg_likes:.2f}")
    print(f"  Average replies per comment: {avg_replies:.2f}")
    
    print()
    
    # Content analysis
    print("Content Analysis:")
    avg_word_count = df['word_count'].mean()
    mentions_percentage = (df['has_mentions'].sum() / len(df)) * 100
    hashtags_percentage = (df['has_hashtags'].sum() / len(df)) * 100
    
    print(f"  Average word count: {avg_word_count:.1f}")
    print(f"  Comments with mentions: {mentions_percentage:.1f}%")
    print(f"  Comments with hashtags: {hashtags_percentage:.1f}%")
    
    print()
    
    # Sentiment analysis (if available)
    if 'sentiment' in df.columns:
        print("Sentiment Analysis:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Top authors by engagement
    print("Top 10 Authors by Likes:")
    top_authors = df.groupby('author')['likes_count'].sum().sort_values(ascending=False).head(10)
    for author, total_likes in top_authors.items():
        comment_count = len(df[df['author'] == author])
        avg_likes_per_comment = total_likes / comment_count
        print(f"  {author}: {total_likes} total likes ({avg_likes_per_comment:.1f} avg) - {comment_count} comments")

def filter_high_engagement_comments(csv_file_path: str, min_likes: int = 10):
    """Filter and export high-engagement comments"""
    
    df = pd.read_csv(csv_file_path)
    
    # Filter high engagement
    high_engagement = df[df['likes_count'] >= min_likes]
    
    print(f"Found {len(high_engagement)} comments with {min_likes}+ likes")
    
    # Sort by engagement
    high_engagement = high_engagement.sort_values(['likes_count', 'replies_count'], ascending=False)
    
    # Save filtered results
    output_file = f"high_engagement_comments_{min_likes}plus.csv"
    high_engagement.to_csv(output_file, index=False)
    
    print(f"High engagement comments saved to: {output_file}")
    
    return high_engagement

def sentiment_analysis_report(csv_file_path: str):
    """Generate detailed sentiment analysis report"""
    
    df = pd.read_csv(csv_file_path)
    
    if 'sentiment' not in df.columns:
        print("Sentiment data not available in this CSV")
        return
    
    # Sentiment by platform
    sentiment_by_platform = df.groupby(['platform', 'sentiment']).size().unstack(fill_value=0)
    print("Sentiment by Platform:")
    print(sentiment_by_platform)
    print()
    
    # Correlation between sentiment and engagement
    sentiment_engagement = df.groupby('sentiment').agg({
        'likes_count': ['mean', 'median', 'sum'],
        'replies_count': ['mean', 'median', 'sum'],
        'word_count': 'mean'
    }).round(2)
    
    print("Sentiment vs Engagement:")
    print(sentiment_engagement)
    print()
    
    # Most engaging positive/negative comments
    positive_comments = df[df['sentiment'] == 'positive'].nlargest(5, 'likes_count')
    negative_comments = df[df['sentiment'] == 'negative'].nlargest(5, 'likes_count')
    
    print("Top Positive Comments:")
    for _, comment in positive_comments.iterrows():
        print(f"  Likes: {comment['likes_count']} | {comment['text'][:100]}...")
    
    print("\nTop Negative Comments:")
    for _, comment in negative_comments.iterrows():
        print(f"  Likes: {comment['likes_count']} | {comment['text'][:100]}...")

def export_filtered_data(csv_file_path: str, filters: Dict):
    """Export data based on custom filters"""
    
    df = pd.read_csv(csv_file_path)
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'platform' in filters:
        filtered_df = filtered_df[filtered_df['platform'].isin(filters['platform'])]
    
    if 'min_likes' in filters:
        filtered_df = filtered_df[filtered_df['likes_count'] >= filters['min_likes']]
    
    if 'min_word_count' in filters:
        filtered_df = filtered_df[filtered_df['word_count'] >= filters['min_word_count']]
    
    if 'sentiment' in filters:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(filters['sentiment'])]
    
    if 'has_mentions' in filters:
        filtered_df = filtered_df[filtered_df['has_mentions'] == filters['has_mentions']]
    
    if 'has_hashtags' in filters:
        filtered_df = filtered_df[filtered_df['has_hashtags'] == filters['has_hashtags']]
    
    # Export filtered data
    output_file = "filtered_comments.csv"
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtered {len(filtered_df)} comments (from {len(df)} total)")
    print(f"Results saved to: {output_file}")
    
    return filtered_df

# Example usage
if __name__ == "__main__":
    # Analyze results from a CSV file
    # analyze_csv_results("comments_My_Business_20240101_123456_abc123.csv")
    
    # Filter high engagement comments
    # high_engagement = filter_high_engagement_comments("comments.csv", min_likes=5)
    
    # Generate sentiment report
    # sentiment_analysis_report("comments.csv")
    
    # Custom filtering example
    custom_filters = {
        'platform': ['youtube', 'instagram'],
        'min_likes': 3,
        'sentiment': ['positive'],
        'has_hashtags': True
    }
    # filtered_data = export_filtered_data("comments.csv", custom_filters)
    
    print("Advanced filtering examples ready to use!")

