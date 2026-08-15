"""Web UI route.

Serves the single page that drives the pipeline from a browser. The page is a
plain client of ``POST /recognize``; nothing here touches the pipeline.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

# Resolved from this module, not the working directory, so the app serves the
# page whatever directory uvicorn was started from.
WEB_ROOT = Path(__file__).resolve().parents[2] / "web" / "static"

router = APIRouter(tags=["web"])


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the upload page.

    Excluded from the OpenAPI schema: this is a page, not part of the API
    contract that Phases 1-6 established.

    Returns:
        The static ``index.html``.
    """
    return FileResponse(WEB_ROOT / "index.html")
