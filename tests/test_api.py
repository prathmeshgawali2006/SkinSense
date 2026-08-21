from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_rejects_non_image():
    fake_file = ("test.txt", b"not an image", "text/plain")
    response = client.post("/predict", files={"file": fake_file})
    assert response.status_code == 400

def test_predict_accepts_valid_image():
    with open("tests/fixtures/sample_acne.jpg", "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert body["predicted_class"] in ["acne", "blackheades", "dark spots", "pores", "wrinkles"]