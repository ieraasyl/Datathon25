"""
Basic usage examples for the Social Media Scraper
"""

import requests
import time
import json

# API base URL
BASE_URL = "http://localhost:8000"

def example_single_platform_scraping():
    """Example: Scrape comments from YouTube videos"""
    
    request_data = {
        "urls": [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0"
        ],
        "business_name": "Rick Astley Official",
        "max_comments_per_url": 100,
        "export_format": "detailed"
    }
    
    # Start scraping job
    response = requests.post(f"{BASE_URL}/scrape/batch", json=request_data)
    job_data = response.json()
    job_id = job_data["job_id"]
    
    print(f"Started job: {job_id}")
    
    # Poll for completion
    while True:
        status_response = requests.get(f"{BASE_URL}/scrape/status/{job_id}")
        status = status_response.json()
        
        print(f"Status: {status['status']}")
        print(f"Progress: {status['progress']}")
        
        if status["status"] == "completed":
            # Download results
            download_response = requests.get(f"{BASE_URL}/scrape/download/{job_id}")
            with open("rick_astley_comments.csv", "wb") as f:
                f.write(download_response.content)
            print("Results downloaded to rick_astley_comments.csv")
            break
        elif status["status"] == "failed":
            print(f"Job failed: {status.get('error_log')}")
            break
        
        time.sleep(5)

def example_multi_platform_scraping():
    """Example: Scrape from multiple platforms"""
    
    request_data = {
        "urls": [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.facebook.com/posts/123456789",
            "https://www.instagram.com/p/ABC123DEF456/",
            "https://vk.com/wall-12345_67890"
        ],
        "business_name": "Multi Platform Brand",
        "max_comments_per_url": 200,
        "export_format": "detailed"
    }
    
    response = requests.post(f"{BASE_URL}/scrape/batch", json=request_data)
    job_data = response.json()
    job_id = job_data["job_id"]
    
    print(f"Started multi-platform job: {job_id}")
    
    # Monitor progress
    while True:
        status_response = requests.get(f"{BASE_URL}/scrape/status/{job_id}")
        status = status_response.json()
        
        progress = status['progress']
        print(f"Processed: {progress['processed']}/{progress['total']} URLs")
        print(f"Comments collected: {progress.get('comments_collected', 0)}")
        
        if status["status"] == "completed":
            download_response = requests.get(f"{BASE_URL}/scrape/download/{job_id}")
            with open("multi_platform_comments.csv", "wb") as f:
                f.write(download_response.content)
            print("Multi-platform results downloaded")
            break
        elif status["status"] == "failed":
            print(f"Job failed: {status.get('error_log')}")
            break
        
        time.sleep(10)

def check_platform_availability():
    """Check which platforms are available"""
    
    response = requests.get(f"{BASE_URL}/platforms")
    platforms_info = response.json()
    
    print("Platform Availability:")
    for platform_name, info in platforms_info["platforms"].items():
        status = "✅ Available" if info["available"] else "❌ Not Available"
        print(f"  {info['name']}: {status}")
        if not info["available"]:
            print(f"    Requires: {info['requires']}")

if __name__ == "__main__":
    print("Social Media Scraper - Basic Usage Examples")
    print("=" * 50)
    
    # Check what's available
    check_platform_availability()
    print()
    
    # Run examples (uncomment as needed)
    # example_single_platform_scraping()
    # example_multi_platform_scraping()
