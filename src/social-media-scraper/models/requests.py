from pydantic import BaseModel, validator
from typing import List, Optional
from enum import Enum

class PlatformEnum(str, Enum):
    YOUTUBE = "youtube"
    VK = "vk"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"

class ScrapeRequest(BaseModel):
    urls: List[str]
    business_name: Optional[str] = "Unknown Business"
    max_comments_per_url: Optional[int] = 500
    platforms: Optional[List[PlatformEnum]] = None
    include_metadata: Optional[bool] = True
    export_format: Optional[str] = "detailed"

    @validator('urls')
    def validate_urls(cls, v):
        if not v:
            raise ValueError('At least one URL is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 URLs allowed per request')
        return v