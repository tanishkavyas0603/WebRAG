import hashlib
import httpx
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urlparse
import ipaddress
import socket
from pathlib import Path

from sqlalchemy.orm import Session
from app.models.db import Document
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class IngestionError(Exception):
    pass

class SSRFProtectionError(IngestionError):
    pass

class DocumentIngestionService:
    def __init__(self, db: Session, user_id: int, url: str):
        self.db = db
        self.user_id = user_id
        self.url = url
        self.max_size_bytes = 10 * 1024 * 1024  # 10 MB limit
        self.timeout = 15.0

    def _validate_url_safety(self, url: str):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise SSRFProtectionError(f"Unsupported scheme: {parsed.scheme}")

        hostname = parsed.hostname
        if not hostname:
            raise SSRFProtectionError("Invalid URL format")

        if hostname.lower() in ("localhost", "127.0.0.1", "[::1]"):
            raise SSRFProtectionError("Localhost is not allowed")

        try:
            ip_addr = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise SSRFProtectionError("Private IP addresses are not allowed")
        except socket.gaierror:
            # Let the actual request handle DNS failures, but it's safe from SSRF if it doesn't resolve
            pass

    async def _validate_request_hook(self, request: httpx.Request):
        """Hook called before EVERY request, including redirects"""
        self._validate_url_safety(str(request.url))

    async def fetch_html(self) -> str:
        logger.info(f"Downloading from {self.url}...")
        
        headers = {
            "User-Agent": settings.WEBRAG_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                max_redirects=5,
                follow_redirects=True,
                event_hooks={'request': [self._validate_request_hook]},
                headers=headers
            ) as client:
                response = await client.get(self.url)
                
                # Check Content-Type early
                content_type = response.headers.get('Content-Type', '').lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type and "text/plain" not in content_type:
                    raise IngestionError("The provided URL does not contain an HTML webpage.")

                response.raise_for_status()
                
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > self.max_size_bytes:
                    raise IngestionError("Document exceeds maximum size limit (10MB).")
                
                text = response.text
                if len(text.encode('utf-8')) > self.max_size_bytes:
                    raise IngestionError("Document exceeds maximum size limit (10MB).")
                    
                logger.info("Download completed.")
                return text

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 403:
                raise IngestionError("This webpage refused automated access (HTTP 403). Try another publicly accessible webpage.")
            elif status == 404:
                raise IngestionError("The webpage could not be found (HTTP 404).")
            elif status == 429:
                raise IngestionError("The webpage is temporarily rate-limiting automated requests.")
            else:
                raise IngestionError(f"HTTP error occurred: {status}")
        except httpx.TimeoutException:
            raise IngestionError("The webpage took too long to respond.")
        except SSRFProtectionError:
            raise # Propagate SSRF specific errors directly
        except httpx.RequestError as e:
            raise IngestionError(f"Failed to connect to the webpage.")


    def extract_text(self, html: str) -> tuple[str, str]:
        logger.info("Parsing HTML...")
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else "Untitled Webpage"
        
        # Method 1: Trafilatura (Primary)
        # We try to extract only the main content body
        clean_text = None
        method_used = "None"
        
        try:
            # Trafilatura favors the main content and excludes sidebars, navigation, footers
            clean_text = trafilatura.extract(html, include_links=False, include_images=False, include_tables=True)
            if clean_text:
                clean_text = clean_text.strip()
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            
        # Validate Trafilatura extraction
        # If it's too short (e.g. under 200 chars), it might have missed the actual content, so we fallback
        if clean_text and len(clean_text) > 200:
            method_used = "trafilatura"
        else:
            # Method 2: BeautifulSoup (Fallback)
            logger.info("Trafilatura extraction was empty or too short. Using BeautifulSoup fallback.")
            method_used = "beautifulsoup"
            
            # Remove a wider range of non-content noisy tags
            for tag in soup(["script", "style", "footer", "nav", "noscript", "iframe", "svg", "form", "header", "aside"]):
                tag.decompose()

            # Remove obvious navigation/sidebar containers by class/id patterns if we wanted to be strict,
            # but standard decompose() covers most of it.
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)
            
        if not clean_text:
            raise IngestionError("No readable content could be extracted from the webpage.")

        logger.info(f"Extraction Diagnostics:")
        logger.info(f"  URL: {self.url}")
        logger.info(f"  Raw HTML length: {len(html)}")
        logger.info(f"  Extracted text length: {len(clean_text)}")
        logger.info(f"  Method used: {method_used}")
        
        # The number of paragraphs can be roughly estimated by double newlines
        paragraphs = [p for p in clean_text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            # If trafilatura returned single newlines mostly, count those
            paragraphs = [p for p in clean_text.split('\n') if p.strip()]
        logger.info(f"  Number of paragraphs: {len(paragraphs)}")
        
        return clean_text, title

    async def process(self) -> Document:
        html = await self.fetch_html()
        clean_text, title = self.extract_text(html)
        
        # Calculate content hash for deduplication
        content_hash = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
        
        # Check if identical document already exists for this user
        existing_doc = self.db.query(Document).filter(
            Document.user_id == self.user_id,
            Document.content_hash == content_hash
        ).first()
        
        if existing_doc:
            logger.info("Identical document already exists, skipping ingestion.")
            return existing_doc
            
        # Create new document record
        document = Document(
            user_id=self.user_id,
            url=self.url,
            title=title[:255], # Truncate if too long
            content_hash=content_hash,
            content=clean_text,
            status="processing"
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        
        return document