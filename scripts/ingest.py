from app.services.ingestion_service import DocumentIngestionService

URL = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2"

if __name__ == "__main__":
    service = DocumentIngestionService(URL)
    output = service.run()

    print(f"\nDocument saved to:\n{output}")