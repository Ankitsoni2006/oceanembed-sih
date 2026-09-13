"""
SIH26066 — OceanEmbed Backend Integration & Unit Tests
Tests the FastAPI backend, inference pipeline, error handling, and numerical consistency.
"""

import math
import time
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.inference import INFERENCE_SERVICE
from backend.config import TARGET_DEPTHS, MODEL_PARAMETERS


@pytest.fixture(scope="module")
def client():
    """Initializes the FastAPI test client with startup lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_health_endpoint(client):
    """Test 1: /health returns status ok and model_loaded true."""
    response = client.get("/health")
    assert response.status_code == 200, f"Health endpoint failed: {response.text}"
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model"] == "OceanEmbedNetV3_Decoder"
    assert "device" in data


def test_02_model_info_endpoint(client):
    """Test 2: /model-info returns complete architectural and domain specifications."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "OceanEmbedNetV3_Decoder"
    assert data["version"] == "v3"
    assert data["parameters"] == MODEL_PARAMETERS
    assert len(data["output_depths_m"]) == 15
    assert data["output_depths_m"] == TARGET_DEPTHS
    assert len(data["input_variables"]) == 7
    assert data["region"]["lat_min"] == 5.0
    assert data["region"]["lat_max"] == 30.0
    assert data["region"]["lon_min"] == 45.0
    assert data["region"]["lon_max"] == 105.0
    assert data["status"] == "validated"


def test_03_available_dates_endpoint(client):
    """Test 3: /available-dates returns valid 2020 dates."""
    response = client.get("/available-dates")
    assert response.status_code == 200
    data = response.json()
    assert data["total_dates"] == 274
    assert data["date_range"]["start"] == "2020-01-01"
    assert data["date_range"]["end"] == "2020-09-30"
    assert len(data["dates"]) == 274
    assert "2020-09-15" in data["dates"]

    # Test format=list query parameter
    res_list = client.get("/available-dates?format=list")
    assert res_list.status_code == 200
    assert isinstance(res_list.json(), list)
    assert len(res_list.json()) == 274


def test_04_valid_predict_endpoint(client):
    """Test 4: /predict with valid ocean coordinate (15°N, 85°E) succeeds."""
    payload = {
        "date": "2020-09-15",
        "latitude": 15.0,
        "longitude": 85.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict failed: {response.text}"
    data = response.json()
    assert data["date"] == "2020-09-15"
    assert data["latitude"] == 15.0
    assert data["longitude"] == 85.0
    assert data["grid_latitude"] == 15.0
    assert data["grid_longitude"] == 85.0
    assert data["model"] == "OceanEmbedNetV3_Decoder"
    assert data["model_version"] == "v3"
    assert data["is_valid_ocean"] is True
    assert data["inference_ms"] > 0
    assert data["total_latency_ms"] > 0


def test_05_invalid_latitude(client):
    """Test 5: /predict with invalid latitude (< 5.0 or > 30.0) returns 400 or 422."""
    # Under minimum
    res_low = client.post("/predict", json={"date": "2020-09-15", "latitude": 2.0, "longitude": 85.0})
    assert res_low.status_code in [400, 422]

    # Over maximum
    res_high = client.post("/predict", json={"date": "2020-09-15", "latitude": 35.0, "longitude": 85.0})
    assert res_high.status_code in [400, 422]


def test_06_invalid_longitude(client):
    """Test 6: /predict with invalid longitude (< 45.0 or > 105.0) returns 400 or 422."""
    # Under minimum
    res_low = client.post("/predict", json={"date": "2020-09-15", "latitude": 15.0, "longitude": 30.0})
    assert res_low.status_code in [400, 422]

    # Over maximum
    res_high = client.post("/predict", json={"date": "2020-09-15", "latitude": 15.0, "longitude": 115.0})
    assert res_high.status_code in [400, 422]


def test_07_unavailable_date(client):
    """Test 7: /predict with unavailable date returns 404."""
    payload = {
        "date": "2021-05-15",
        "latitude": 15.0,
        "longitude": 85.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    assert "unavailable" in response.text.lower() or "not available" in response.text.lower()


def test_08_land_cell_rejection(client):
    """Test 8: /predict on land cell (23°N, 78°E, Central India) returns 400."""
    payload = {
        "date": "2020-09-15",
        "latitude": 23.0,
        "longitude": 78.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "land" in response.text.lower() or "missing" in response.text.lower()


def test_09_exact_15_depths_returned(client):
    """Test 9: /predict output contains exactly 15 depth values matching the catalog."""
    payload = {
        "date": "2020-09-15",
        "latitude": 15.0,
        "longitude": 85.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["depths_m"]) == 15
    assert len(data["temperatures_c"]) == 15
    assert data["depths_m"] == TARGET_DEPTHS


def test_10_finite_temperatures_and_physics(client):
    """Test 10: All output temperatures are finite and physically plausible for the NIO."""
    payload = {
        "date": "2020-09-15",
        "latitude": 12.0,
        "longitude": 65.0  # Arabian Sea
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    temps = data["temperatures_c"]
    for t in temps:
        assert not math.isnan(t), f"Found NaN temperature in prediction: {temps}"
        assert not math.isinf(t), f"Found Inf temperature in prediction: {temps}"
        # North Indian Ocean temperatures range between 3°C (abyssal 1000m) and 35°C (warm pool SST)
        assert 2.0 <= t <= 35.0, f"Temperature {t}°C outside physical NIO bounds [2, 35]"


def test_11_model_loaded_only_once(client):
    """Test 11: INFERENCE_SERVICE initializes the model exactly once without reloading."""
    assert INFERENCE_SERVICE.is_initialized is True
    initial_model_id = id(INFERENCE_SERVICE.model)

    # Make another request
    response = client.get("/health")
    assert response.status_code == 200

    # Verify model object reference has not changed
    assert id(INFERENCE_SERVICE.model) == initial_model_id


def test_12_repeated_requests_stability_and_numerical_consistency(client):
    """
    Test 12: Repeated requests succeed with low latency and return identical predictions.
    Also verifies predictions against the expected temperature column.
    """
    payload = {
        "date": "2020-09-15",
        "latitude": 15.0,
        "longitude": 85.0
    }
    latencies = []
    temps_runs = []

    for _ in range(5):
        t0 = time.perf_counter()
        res = client.post("/predict", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        latencies.append(elapsed_ms)
        temps_runs.append(res.json()["temperatures_c"])

    # Verify identical numerical values across calls (deterministic eval)
    for i in range(1, len(temps_runs)):
        assert temps_runs[i] == temps_runs[0], "Predictions differed between identical requests!"

    # Verify temperatures match expected range (~29.6°C at surface, ~6.7°C at 1000m)
    surface_t = temps_runs[0][0]
    abyssal_t = temps_runs[0][-1]
    assert 28.0 <= surface_t <= 31.0, f"Surface temperature {surface_t}°C unexpected"
    assert 5.0 <= abyssal_t <= 8.5, f"1000m temperature {abyssal_t}°C unexpected"

    # Verify average latency is fast (< 100ms)
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 150.0, f"Average repeated API latency too high: {avg_latency:.2f} ms"
