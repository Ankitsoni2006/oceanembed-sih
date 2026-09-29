"""
SIH26066 - OceanEmbed Interactive ARGO Observational Evaluation Tests
====================================================================

Verifies the interactive ARGO evaluation API surface and the scientific
integrity of the aggregate result.

Scientific framing honoured by these tests:
* ARGO is an EVALUATION-ONLY source; it never enters the model as an input.
* Every metric asserted here is computed by the backend from authentic ARGO
  observations and fresh forward passes of the frozen model.
* The one hardcoded number is the published regression target (0.7973 degC),
  which is exactly where such a target belongs - in a test, never in the API.
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import TARGET_DEPTHS
from backend.argo import ARGO_VALIDATION_SERVICE

# ---------------------------------------------------------------------------
# Authentic dataset composition for the fixed September 2020 evaluation archive
# ---------------------------------------------------------------------------
EXPECTED_PROFILE_COUNT = 36
EXPECTED_UNIQUE_WMO_COUNT = 28
EXPECTED_MATCHED_OBSERVATIONS = 497

# ---------------------------------------------------------------------------
# Published regression target for the aggregate OceanEmbed v3 ARGO evaluation
# ---------------------------------------------------------------------------
PUBLISHED_ARGO_RMSE_C = 0.7973
ARGO_RMSE_TOLERANCE_C = 0.05   # ~6% band: catches model/scaler/matching regressions,
                               # tolerates library-level numerical noise


@pytest.fixture(scope="module")
def client():
    """Initializes the FastAPI test client with startup lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def profiles_payload(client):
    response = client.get("/argo/profiles")
    assert response.status_code == 200, f"/argo/profiles failed: {response.text}"
    return response.json()


@pytest.fixture(scope="module")
def summary_payload(client):
    response = client.get("/argo/summary")
    assert response.status_code == 200, f"/argo/summary failed: {response.text}"
    return response.json()


# ---------------------------------------------------------------------------
# 1. Catalog availability
# ---------------------------------------------------------------------------
def test_01_argo_profiles_returns_authentic_profile_count(profiles_payload):
    """Test 1: /argo/profiles returns the 36 authentic ARGO profiles."""
    assert profiles_payload["total_profiles"] == EXPECTED_PROFILE_COUNT
    assert len(profiles_payload["profiles"]) == EXPECTED_PROFILE_COUNT
    assert profiles_payload["evaluation_type"] == "INDEPENDENT_OFFLINE_ARGO_OBSERVATIONAL_EVALUATION"


def test_02_twenty_eight_unique_wmo_floats_are_represented(profiles_payload):
    """Test 2: 28 unique WMO profiling floats are represented."""
    wmo_ids = [p["wmo"] for p in profiles_payload["profiles"]]
    assert len(set(wmo_ids)) == EXPECTED_UNIQUE_WMO_COUNT
    assert profiles_payload["unique_wmo_count"] == EXPECTED_UNIQUE_WMO_COUNT


def test_03_profile_ids_are_unique(profiles_payload):
    """Test 3: every profile_id is unique and non-empty."""
    ids = [p["profile_id"] for p in profiles_payload["profiles"]]
    assert len(ids) == len(set(ids)), "Duplicate profile_id values found in the ARGO catalog"
    assert all(isinstance(pid, str) and pid.strip() for pid in ids)


def test_04_all_profiles_have_valid_coordinates(profiles_payload):
    """Test 4: every profile lies inside the SIH North Indian Ocean domain."""
    for p in profiles_payload["profiles"]:
        assert isinstance(p["latitude"], float)
        assert isinstance(p["longitude"], float)
        assert not math.isnan(p["latitude"]) and not math.isnan(p["longitude"])
        assert 5.0 <= p["latitude"] <= 30.0, f"{p['profile_id']} latitude out of domain: {p['latitude']}"
        assert 45.0 <= p["longitude"] <= 105.0, f"{p['profile_id']} longitude out of domain: {p['longitude']}"


def test_04b_wmo_identifiers_are_correctly_decoded(profiles_payload):
    """
    Test 4b: WMO identifiers are decoded as characters, not as raw byte values.

    Regression guard for the NetCDF char-decoding defect that produced values
    such as '5057485049555232' in the earlier offline matching ledger.
    """
    for p in profiles_payload["profiles"]:
        wmo = p["wmo"]
        assert wmo.isdigit(), f"WMO identifier is not numeric: {wmo!r}"
        assert 6 <= len(wmo) <= 8, f"WMO identifier has implausible length: {wmo!r}"


def test_04c_observation_dates_are_the_authentic_snapshot_dates(profiles_payload):
    """Test 4c: the evaluation set covers exactly the three authentic snapshot days."""
    assert profiles_payload["observation_dates"] == ["2020-09-01", "2020-09-15", "2020-09-25"]
    dates = {p["date"] for p in profiles_payload["profiles"]}
    assert dates == {"2020-09-01", "2020-09-15", "2020-09-25"}
