"""Schemas for the health endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Attributes:
        status: Liveness indicator; ``"ok"`` when the service is healthy.
        service: The service name.
        version: The running application version.
    """

    status: str
    service: str
    version: str
