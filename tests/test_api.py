import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "ml-workspace-ai"


def test_create_project():
    """Test project creation."""
    response = client.post(
        "/api/projects",
        json={"name": "Test Project", "description": "A test project"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Project"
    assert data["description"] == "A test project"
    assert "id" in data


def test_list_projects():
    """Test listing projects."""
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_create_conversation():
    """Test conversation creation."""
    response = client.post(
        "/api/conversations",
        json={"title": "Test Conversation"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Conversation"
    assert "id" in data


def test_list_conversations():
    """Test listing conversations."""
    response = client.get("/api/conversations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_search_without_documents():
    """Test search endpoint with no documents."""
    response = client.post(
        "/api/search",
        json={"query": "test query", "top_k": 5}
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)
