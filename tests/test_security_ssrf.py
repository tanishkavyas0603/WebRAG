import pytest
import asyncio
from app.services.ingestion_service import DocumentIngestionService, SSRFProtectionError, IngestionError

# Tests for _validate_url_safety directly

@pytest.fixture
def ingestion_service():
    # Pass None for db and user_id since _validate_url_safety doesn't need them
    return DocumentIngestionService(db=None, user_id=1, url="http://example.com")


def test_ssrf_rejects_localhost_string(ingestion_service):
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("http://localhost:8000")
        
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("http://127.0.0.1")

    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("http://[::1]")
        
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("http://0.0.0.0")

def test_ssrf_rejects_unsupported_protocols(ingestion_service):
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("file:///etc/passwd")
        
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("ftp://example.com")
        
    with pytest.raises(SSRFProtectionError):
        ingestion_service._validate_url_safety("gopher://example.com")


def test_ssrf_redirect_to_localhost_blocked():
    # To test redirect handling without needing an actual malicious server, 
    # we can mock httpx.AsyncClient.get to simulate a redirect to localhost
    import httpx
    
    class MockResponse:
        def __init__(self, status_code, location=None, content_type="text/html", text="Mocked text"):
            self.status_code = status_code
            self.headers = {"Content-Type": content_type}
            if location:
                self.headers["Location"] = location
            self.text = text
            self.url = "http://initial.com"
            
        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("Error", request=None, response=self)

    service = DocumentIngestionService(db=None, user_id=1, url="http://initial.com")
    
    async def mock_fetch(service):
        service.timeout = 1
        current_url = service.url
        redirects = 0
        
        while redirects <= 5:
            service._validate_url_safety(current_url)
            
            # Simulate the client get
            if "initial.com" in current_url:
                response = MockResponse(302, location="http://127.0.0.1/admin")
            elif "127.0.0.1" in current_url:
                response = MockResponse(200, text="Secret")
                
            if response.status_code in (301, 302, 303, 307, 308):
                redirect_url = response.headers.get("Location")
                current_url = redirect_url
                redirects += 1
                continue
                
            return response.text
            
    # We should get an SSRF error when the redirect loop reaches the localhost url
    with pytest.raises(SSRFProtectionError):
        asyncio.run(mock_fetch(service))

