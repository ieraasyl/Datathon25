from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
import logging
import uuid

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        logger.info(f"Request {request_id}: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log request completion
        process_time = time.time() - start_time
        logger.info(f"Request {request_id}: Completed in {process_time:.3f}s - Status: {response.status_code}")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = {}  # In production, use Redis
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        cutoff_time = current_time - 60
        self.requests = {
            ip: timestamps for ip, timestamps in self.requests.items()
            if any(t > cutoff_time for t in timestamps)
        }
        
        # Update timestamps for current IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove old timestamps for this IP
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > cutoff_time
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.calls_per_minute:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Add current request timestamp
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)