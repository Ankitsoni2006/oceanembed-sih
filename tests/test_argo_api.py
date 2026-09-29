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


def test_04d_catalog_observation_total(profiles_payload):
    """Test 4d: the catalog carries 497 genuine profile-depth observations."""
    assert profiles_payload["total_valid_observations"] == EXPECTED_MATCHED_OBSERVATIONS
    total = sum(p["valid_target_depth_count"] for p in profiles_payload["profiles"])
    assert total == EXPECTED_MATCHED_OBSERVATIONS
    for p in profiles_payload["profiles"]:
        assert len(p["observed_depths_m"]) == p["valid_target_depth_count"]
        assert set(p["observed_depths_m"]) <= set(TARGET_DEPTHS)


# ---------------------------------------------------------------------------
# 2. Aggregate evaluation (/argo/summary)
# ---------------------------------------------------------------------------
def test_05_summary_counts(summary_payload):
    """Test 5: the aggregate covers every authentic profile, float and observation."""
    assert summary_payload["profile_count"] == EXPECTED_PROFILE_COUNT
    assert summary_payload["unique_wmo_count"] == EXPECTED_UNIQUE_WMO_COUNT
    assert summary_payload["matched_observation_count"] == EXPECTED_MATCHED_OBSERVATIONS
    assert summary_payload["skipped_profiles"] == []


def test_06_summary_reproduces_published_argo_rmse(summary_payload):
    """Test 6: the request-time aggregate RMSE reproduces the published benchmark."""
    rmse = summary_payload["rmse"]
    assert rmse is not None and math.isfinite(rmse)
    assert abs(rmse - PUBLISHED_ARGO_RMSE_C) <= ARGO_RMSE_TOLERANCE_C, (
        f"Aggregate ARGO RMSE {rmse} degC deviates from published {PUBLISHED_ARGO_RMSE_C} degC"
    )
    # Internal consistency of the dynamically computed statistics
    assert 0 < summary_payload["mae"] <= rmse
    assert abs(summary_payload["bias"]) <= summary_payload["mae"]


def test_07_summary_depth_wise_counts_sum_to_total(summary_payload):
    """Test 7: depth-wise counts partition the matched observations exactly."""
    rows = summary_payload["depth_wise"]
    assert [r["depth_m"] for r in rows] == TARGET_DEPTHS
    assert sum(r["count"] for r in rows) == summary_payload["matched_observation_count"]
    for r in rows:
        if r["count"] == 0:
            # Missing observations stay missing: no statistic is invented for an empty depth.
            assert r["rmse"] is None and r["mae"] is None and r["bias"] is None


def test_08_summary_reports_degraded_input_cells(summary_payload):
    """Test 8: profiles on cells with incomplete surface input are reported, not dropped."""
    meta = summary_payload["profile_metadata"]
    degraded_ids = {p["profile_id"] for p in meta["profiles_with_degraded_surface_inputs"]}
    assert set(meta["profiles_mapped_to_non_ocean_cells"]) <= degraded_ids
    for p in meta["profiles_with_degraded_surface_inputs"]:
        assert p["available_count"] < 7
        assert len(p["missing_channels"]) == 7 - p["available_count"]


# ---------------------------------------------------------------------------
# 3. Single-profile comparison (/argo/compare/{profile_id})
# ---------------------------------------------------------------------------
def test_09_compare_every_profile(client, profiles_payload):
    """Test 9: every catalog profile can be compared, and only observed depths are scored."""
    total_pairs = 0
    for p in profiles_payload["profiles"]:
        response = client.get(f"/argo/compare/{p['profile_id']}")
        assert response.status_code == 200, f"{p['profile_id']}: {response.text}"
        data = response.json()

        assert data["argo_profile"]["profile_id"] == p["profile_id"]
        assert data["argo_profile"]["wmo"] == p["wmo"]
        assert data["matched_model_date"] == p["date"]
        assert data["depths_m"] == TARGET_DEPTHS
        assert len(data["depth_comparison"]) == len(TARGET_DEPTHS)

        observed_rows = [r for r in data["depth_comparison"] if r["observed"]]
        assert [r["depth_m"] for r in observed_rows] == p["observed_depths_m"]
        for r in data["depth_comparison"]:
            assert r["predicted_c"] is not None and math.isfinite(r["predicted_c"])
            if r["observed"]:
                assert r["observed_c"] is not None
                assert math.isclose(r["error_c"], r["predicted_c"] - r["observed_c"], abs_tol=2e-4)
            else:
                # An unobserved depth is never filled and never contributes an error.
                assert r["observed_c"] is None and r["error_c"] is None

        assert data["metrics"]["count"] == len(observed_rows)
        errors = [r["error_c"] for r in observed_rows]
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        assert math.isclose(data["metrics"]["rmse"], rmse, abs_tol=1e-3)
        assert data["spatial_offset_km"] < 20.0
        total_pairs += data["metrics"]["count"]

    assert total_pairs == EXPECTED_MATCHED_OBSERVATIONS


def test_10_compare_prediction_equals_surface_only_reconstruction(client, profiles_payload):
    """
    Test 10: the ARGO comparison uses exactly the surface-only /predict reconstruction.

    /predict never receives ARGO data, so an identical 15-depth column proves the
    float's observations play no part in the model input.
    """
    checked = 0
    for p in profiles_payload["profiles"]:
        cmp_data = client.get(f"/argo/compare/{p['profile_id']}").json()
        if not cmp_data["is_valid_ocean"]:
            continue  # /predict rejects cells without a valid SST mask by design
        pred = client.post(
            "/predict",
            json={"date": cmp_data["matched_model_date"],
                  "latitude": cmp_data["grid_latitude"],
                  "longitude": cmp_data["grid_longitude"]},
        )
        assert pred.status_code == 200, pred.text
        compared = [r["predicted_c"] for r in cmp_data["depth_comparison"]]
        for a, b in zip(compared, pred.json()["temperatures_c"]):
            assert math.isclose(a, b, abs_tol=1e-3)
        checked += 1
        if checked >= 5:
            break
    assert checked == 5


def test_11_different_profiles_give_different_comparisons(client, profiles_payload):
    """Test 11: selecting another profile yields that profile's own data (no stale reuse)."""
    ids = [p["profile_id"] for p in profiles_payload["profiles"][:2]]
    a = client.get(f"/argo/compare/{ids[0]}").json()
    b = client.get(f"/argo/compare/{ids[1]}").json()
    assert a["argo_profile"]["profile_id"] != b["argo_profile"]["profile_id"]
    assert (a["grid_latitude"], a["grid_longitude"]) != (b["grid_latitude"], b["grid_longitude"])
    assert [r["observed_c"] for r in a["depth_comparison"]] != [r["observed_c"] for r in b["depth_comparison"]]


@pytest.mark.parametrize("bad_id", ["does_not_exist", "20200901_prof.nc_99999", "..%2F..%2Fetc%2Fpasswd", "%00"])
def test_12_unknown_or_malformed_profile_id_returns_404(client, bad_id):
    """Test 12: unknown/malformed IDs return a clean 404 without filesystem details."""
    response = client.get(f"/argo/compare/{bad_id}")
    assert response.status_code == 404
    body = response.text.lower()
    assert "traceback" not in body
    assert "\\" not in body and "c:/" not in body and "/users/" not in body


# ---------------------------------------------------------------------------
# 4. Degraded-archive behaviour
# ---------------------------------------------------------------------------
def test_13_missing_archive_raises_file_not_found(tmp_path):
    """Test 13: an empty ARGO directory is reported, never silently treated as zero profiles."""
    from src.validation.argo_catalog import ArgoCatalog

    with pytest.raises(FileNotFoundError):
        ArgoCatalog(argo_dir=str(tmp_path)).build()


def test_14_archive_failure_returns_503_and_predict_still_works(client, monkeypatch):
    """Test 14: an unavailable ARGO archive yields 503 on ARGO routes; /predict is unaffected."""
    def _unavailable(*_args, **_kwargs):
        raise FileNotFoundError("simulated missing ARGO archive at C:/secret/path")

    monkeypatch.setattr(ARGO_VALIDATION_SERVICE, "compare", _unavailable)
    monkeypatch.setattr(ARGO_VALIDATION_SERVICE, "summary", _unavailable)

    for path in ("/argo/compare/20200901_prof.nc_101", "/argo/summary"):
        response = client.get(path)
        assert response.status_code == 503
        assert "secret" not in response.text  # filesystem details are not exposed

    pred = client.post("/predict", json={"date": "2020-09-15", "latitude": 15.0, "longitude": 85.0})
    assert pred.status_code == 200
