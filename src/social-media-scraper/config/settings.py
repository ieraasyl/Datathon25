import os
from typing import Optional

class Settings:
    """Application settings"""
    
    def __init__(self):
        self.youtube_api_key: Optional[str] = os.getenv("YT_API_KEY")
        self.vk_token: Optional[str] = os.getenv("VK_TOKEN")
        self.facebook_token: Optional[str] = os.getenv("FB_TOKEN")
        self.instagram_token: Optional[str] = os.getenv("IG_TOKEN")
        
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "5"))
        self.output_directory: str = os.getenv("OUTPUT_DIRECTORY", "output")

settings = Settings()