from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
import logging
from .models.requests import ScrapeRequest
from .models.responses import BatchScrapeStatus
from .scrapers.youtube import YouTubeScraper
from .utils.csv_handler import CSVHandler
from .utils.url_parser import URLParser
from .config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Social Comments Scraper",
    description="Batch scraper for social media comments with rich CSV export",
    version="2.0.0"
)

# Global job storage (use Redis in production)
active_jobs = {}

class ScraperManager:
    """Manage different scraper instances"""
    
    def __init__(self):
        self.scrapers = {}
        if settings.youtube_api_key:
            self.scrapers['youtube'] = YouTubeScraper(settings.youtube_api_key)
        # Add other scrapers as needed
    
    def get_scraper(self, platform: str):
        """Get scraper for specific platform"""
        return self.scrapers.get(platform)

scraper_manager = ScraperManager()
csv_handler = CSVHandler(settings.output_directory)

async def process_batch_scraping(job_id: str, request: ScrapeRequest):
    """Background task for batch processing"""
    all_comments = []
    processed_urls = 0
    total_urls = len(request.urls)
    
    active_jobs[job_id]['status'] = 'processing'
    active_jobs[job_id]['progress'] = {'processed': 0, 'total': total_urls}
    
    try:
        for url in request.urls:
            platform = URLParser.detect_platform(url)
            if not platform:
                logger.warning(f"Unknown platform for URL: {url}")
                continue
            
            scraper = scraper_manager.get_scraper(platform)
            if not scraper:
                logger.warning(f"No scraper available for platform: {platform}")
                continue
            
            comments = await scraper.scrape_comments(
                url, request.max_comments_per_url, request.business_name
            )
            
            all_comments.extend(comments)
            processed_urls += 1
            
            active_jobs[job_id]['progress'] = {
                'processed': processed_urls,
                'total': total_urls,
                'comments_collected': len(all_comments)
            }
        
        # Save to CSV
        if all_comments:
            csv_file = csv_handler.save_comments(
                all_comments, request.business_name, request.export_format
            )
            active_jobs[job_id]['results_file'] = csv_file
            active_jobs[job_id]['status'] = 'completed'
        else:
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['error_log'] = 'No comments found'
            
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error_log'] = str(e)

@app.post('/scrape/batch', response_model=BatchScrapeStatus)
async def scrape_batch(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start batch scraping of multiple URLs"""
    job_id = str(uuid.uuid4())
    
    active_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': {'processed': 0, 'total': len(request.urls)},
        'results_file': None,
        'error_log': None
    }
    
    background_tasks.add_task(process_batch_scraping, job_id, request)
    
    return BatchScrapeStatus(**active_jobs[job_id])

@app.get('/scrape/status/{job_id}', response_model=BatchScrapeStatus)
async def get_scrape_status(job_id: str):
    """Get status of a scraping job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return BatchScrapeStatus(**active_jobs[job_id])

@app.get('/scrape/download/{job_id}')
async def download_results(job_id: str):
    """Download CSV results for a completed job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job['status'] != 'completed' or not job['results_file']:
        raise HTTPException(status_code=400, detail="Job not completed or no results available")
    
    filepath = Path(job['results_file'])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=filepath,
        filename=filepath.name,
        media_type='text/csv'
    )

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_platforms": list(scraper_manager.scrapers.keys()),
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    