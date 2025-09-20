import click
import asyncio
import json
from typing import List
from .scrapers.youtube import YouTubeScraper
from .scrapers.facebook import FacebookScraper
from .scrapers.instagram import InstagramScraper
from .scrapers.vk import VKScraper
from .utils.csv_handler import CSVHandler
from .utils.url_parser import URLParser
from .config.settings import settings

@click.group()
def cli():
    """Social Media Comments Scraper CLI"""
    pass

@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--business-name', '-b', default='CLI_Business', help='Business name for the scraped data')
@click.option('--max-comments', '-m', default=500, help='Maximum comments per URL')
@click.option('--output-format', '-f', default='csv', type=click.Choice(['csv', 'json', 'excel']), help='Output format')
@click.option('--output-file', '-o', help='Output filename (without extension)')
def scrape(urls: List[str], business_name: str, max_comments: int, output_format: str, output_file: str):
    """Scrape comments from social media URLs"""
    async def run_scraping():
        all_comments = []
        
        for url in urls:
            platform = URLParser.detect_platform(url)
            if not platform:
                click.echo(f"Warning: Could not detect platform for {url}")
                continue
            
            scraper = None
            try:
                if platform == 'youtube' and settings.youtube_api_key:
                    scraper = YouTubeScraper(settings.youtube_api_key)
                elif platform == 'facebook' and settings.facebook_token:
                    scraper = FacebookScraper(settings.facebook_token)
                elif platform == 'instagram' and settings.instagram_token:
                    scraper = InstagramScraper(settings.instagram_token)
                elif platform == 'vk':
                    scraper = VKScraper(settings.vk_token)
                
                if not scraper:
                    click.echo(f"Warning: No API key configured for {platform}")
                    continue
                
                click.echo(f"Scraping {url}...")
                comments = await scraper.scrape_comments(url, max_comments, business_name)
                all_comments.extend(comments)
                click.echo(f"Collected {len(comments)} comments")
                
            except Exception as e:
                click.echo(f"Error scraping {url}: {str(e)}")
        
        if not all_comments:
            click.echo("No comments were collected")
            return
        
        # Save results
        if not output_file:
            output_file = f"comments_{business_name.replace(' ', '_')}"
        
        if output_format == 'csv':
            csv_handler = CSVHandler()
            result_file = csv_handler.save_comments(all_comments, business_name, 'detailed')
        elif output_format == 'json':
            from .utils.export_formats import DataExporter
            exporter = DataExporter()
            result_file = exporter.to_json(all_comments, output_file)
        elif output_format == 'excel':
            from .utils.export_formats import DataExporter
            exporter = DataExporter()
            result_file = exporter.to_excel(all_comments, output_file)
        
        click.echo(f"Results saved to: {result_file}")
        click.echo(f"Total comments collected: {len(all_comments)}")
    
    asyncio.run(run_scraping())

@cli.command()
def config():
    """Show current configuration"""
    config_info = {
        'YouTube API': 'Configured' if settings.youtube_api_key else 'Not configured',
        'VK Token': 'Configured' if settings.vk_token else 'Not configured',
        'Facebook Token': 'Configured' if settings.facebook_token else 'Not configured',
        'Instagram Token': 'Configured' if settings.instagram_token else 'Not configured',
        'Output Directory': settings.output_directory,
        'Log Level': settings.log_level,
        'Debug Mode': settings.debug
    }
    
    click.echo("Current Configuration:")
    for key, value in config_info.items():
        click.echo(f"  {key}: {value}")

@cli.command()
@click.argument('url')
def validate(url: str):
    """Validate a social media URL"""
    platform = URLParser.detect_platform(url)
    
    if platform:
        click.echo(f"✅ Valid {platform} URL")
        
        # Test if we can extract post ID
        if platform == 'youtube':
            post_id = URLParser.extract_youtube_id(url)
        elif platform == 'vk':
            post_id = URLParser.extract_vk_post(url)
        else:
            post_id = "extraction not implemented"
        
        click.echo(f"   Post ID: {post_id}")
    else:
        click.echo("❌ Invalid or unsupported URL")

if __name__ == '__main__':
    cli()
