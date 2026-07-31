"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a TestClient bound to a freshly built app instance."""
    with TestClient(create_app()) as test_client:
        yield test_client
