"""Tests for the web UI routes.

Only the wiring is asserted: that the page is served and that the static mount
resolves. Markup and script internals are deliberately not asserted — those
tests break on every edit and prove nothing about behaviour.
"""

from fastapi.testclient import TestClient


def test_index_serves_html(client: TestClient) -> None:
    """The root path returns the upload page."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_static_assets_are_mounted(client: TestClient) -> None:
    """The static mount resolves regardless of the process working directory."""
    for asset in ("app.js", "style.css"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200, asset


def test_unknown_static_asset_is_not_found(client: TestClient) -> None:
    """The mount does not serve paths that do not exist."""
    assert client.get("/static/nope.js").status_code == 404


def test_page_is_absent_from_the_api_schema(client: TestClient) -> None:
    """The page is not part of the API contract."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert "/recognize" in paths


def test_mount_does_not_shadow_the_api(client: TestClient) -> None:
    """Existing routes still answer after the UI is mounted."""
    assert client.get("/health").status_code == 200
