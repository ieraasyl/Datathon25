from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import List
import uuid
from pathlib import Path
import logging

from ..models.requests import ScrapeRequest
from ..models.responses import BatchScrapeStatus
from ..utils.csv_handler import CSVHandler
from ..utils.url_parser import URLParser
from ..utils.validators import URLValidator
from .dependencies import get_scraper_factory, ScraperFactory
from ..config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global job storage (use Redis/database in production)
active_jobs = {}

csv_handler = CSVHandler(settings.output_directory)

async def process_batch_scraping(
    job_id: str, 
    request: ScrapeRequest, 
    scraper_factory: ScraperFactory
):
    """Background task for batch processing"""
    all_comments = []
    processed_urls = 0
    total_urls = len(request.urls)
    errors = []
    
    # Update job status
    active_jobs[job_id]['status'] = 'processing'
    active_jobs[job_id]['progress'] = {
        'processed': 0, 
        'total': total_urls,
        'comments_collected': 0,
        'errors': []
    }
    
    try:
        # Validate URLs
        url_validation = URLValidator.validate_batch_urls(request.urls)
        
        if url_validation['invalid']:
            errors.extend([f"Invalid URL: {url}" for url in url_validation['invalid']])
        
        if url_validation['unsupported']:
            errors.extend([f"Unsupported platform: {url}" for url in url_validation['unsupported']])
        
        # Process valid URLs
        for platform, urls in url_validation['valid'].items():
            # Check if platform is available
            if not scraper_factory.is_platform_available(platform):
                error_msg = f"Platform {platform} not available"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
            
            scraper = scraper_factory.get_scraper(platform)
            
            for url in urls:
                try:
                    logger.info(f"Processing URL {processed_urls + 1}/{total_urls}: {url}")
                    
                    comments = await scraper.scrape_comments(
                        url, 
                        request.max_comments_per_url, 
                        request.business_name
                    )
                    
                    all_comments.extend(comments)
                    processed_urls += 1
                    
                    # Update progress
                    active_jobs[job_id]['status'] = 'failed'
        active_jobs[job_id]['error_log'] = error_msg

@router.post('/batch', response_model=BatchScrapeStatus)
async def scrape_batch(
    request: ScrapeRequest, 
    background_tasks: BackgroundTasks,
    scraper_factory: ScraperFactory = Depends(get_scraper_factory)
):
    """Start batch scraping of multiple URLs"""
    job_id = str(uuid.uuid4())
    
    # Validate request
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")
    
    # Initialize job
    active_jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': {
            'processed': 0, 
            'total': len(request.urls),
            'comments_collected': 0,
            'errors': []
        },
        'results_file': None,
        'error_log': None,
        'business_name': request.business_name
    }
    
    # Start background processing
    background_tasks.add_task(
        process_batch_scraping, 
        job_id, 
        request, 
        scraper_factory
    )
    
    logger.info(f"Started batch scraping job {job_id} for {len(request.urls)} URLs")
    
    return BatchScrapeStatus(**active_jobs[job_id])

@router.get('/status/{job_id}', response_model=BatchScrapeStatus)
async def get_scrape_status(job_id: str):
    """Get status of a scraping job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return BatchScrapeStatus(**active_jobs[job_id])

@router.get('/download/{job_id}')
async def download_results(job_id: str):
    """Download CSV results for a completed job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job['status'] != 'completed' or not job['results_file']:
        raise HTTPException(
            status_code=400, 
            detail="Job not completed or no results available"
        )
    
    filepath = Path(job['results_file'])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        path=filepath,
        filename=filepath.name,
        media_type='text/csv'
    )

@router.get('/jobs')
async def list_jobs(limit: int = 50, status_filter: str = None):
    """List recent scraping jobs"""
    jobs = list(active_jobs.values())
    
    # Filter by status if provided
    if status_filter:
        jobs = [job for job in jobs if job['status'] == status_filter]
    
    # Sort by most recent first (assuming job_id contains timestamp info)
    # In production, you'd have created_at timestamp
    jobs.sort(key=lambda x: x['job_id'], reverse=True)
    
    return {
        'jobs': jobs[:limit],
        'total': len(jobs),
        'available_statuses': ['queued', 'processing', 'completed', 'failed']
    }

@router.delete('/jobs/{job_id}')
async def cancel_job(job_id: str):
    """Cancel or delete a scraping job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    # If job is still processing, mark as cancelled
    if job['status'] in ['queued', 'processing']:
        job['status'] = 'cancelled'
        job['error_log'] = 'Job cancelled by user'
    
    # Delete job record
    del active_jobs[job_id]
    
    # Clean up result file if exists
    if job.get('results_file'):
        try:
            Path(job['results_file']).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete result file: {str(e)}")
    
    return {"message": f"Job {job_id} has been cancelled and deleted"}
