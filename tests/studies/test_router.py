"""
Tests for studies module routes
"""

import pytest
from httpx import AsyncClient, ASGITransport

from src.main import app


@pytest.fixture
async def client():
    """Async test client fixture"""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint"""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_studies(client: AsyncClient):
    """Test getting all studies"""
    response = await client.get("/api/studies")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 7  # We have 7 studies


@pytest.mark.asyncio
async def test_get_subjects(client: AsyncClient):
    """Test getting subjects for a study"""
    response = await client.get("/api/study/1/subjects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_invalid_study_number(client: AsyncClient):
    """Test invalid study number returns 400"""
    response = await client.get("/api/study/99/subjects")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_study_stats(client: AsyncClient):
    """Test getting study statistics"""
    response = await client.get("/api/study/1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "totalRecords" in data or "total_records" in data
