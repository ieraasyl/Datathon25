from pydantic import BaseModel
from typing import Optional, Dict

class BatchScrapeStatus(BaseModel):
    job_id: str
    status: str
    progress: Dict
    results_file: Optional[str] = None
    error_log: Optional[str] = None
