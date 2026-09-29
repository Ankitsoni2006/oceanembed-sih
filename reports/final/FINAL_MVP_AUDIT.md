# OceanEmbed — Final MVP Audit (SIH26066)

**Date:** 2026-09-29 · **Branch:** `final-mvp-hardening` (from checkpoint commit `50c7c6b`) · **Not merged to `main`.**
Machine-readable results: [`FINAL_MVP_VERIFICATION.json`](FINAL_MVP_VERIFICATION.json).
Every number in this document was produced by commands run on 2026-09-29 unless it is explicitly
labelled *offline benchmark* (read from the committed experiment reports).

---

## 1. Executive summary

OceanEmbed is a working, scientifically honest research prototype. The frozen v3 model, checkpoint,
scaler and dataset were verified unchanged. The ARGO observational evaluation reproduces exactly
through the live API: **36 profiles, 28 WMO floats, 497 profile-depth observations,
RMSE 0.7973 °C, MAE 0.4820 °C, Bias +0.0507 °C**.

The audit found no scientific violations. ARGO never enters the model input, and missing
observations stay missing. It did find, and fix, engineering and presentation defects:
* a README full of wrong numbers and non-existent endpoints;
* a NumPy-version crash risk in `/predict`;
* an incomplete ARGO test suite, and a pytest configuration that made `pytest -q` error;
* stale reconstruction results shown under new inputs;
* an input field that snapped to defaults when cleared;
* a hardcoded backend URL in the header;
* inconsistent thermocline bands;
* a broken pipeline layout;
* a missing favicon (console 404);
* several over-claiming phrases.

The final state:
* **52/52 pytest** (20/20 ARGO) pass;
* compile, type-check and production build pass;
* all API smoke tests pass on localhost and the LAN IP;
* a scripted browser run of the full judge demo passes **34/34** checks with no console errors other than one deliberate 400.

## 2. System architecture

```
Browser (React 19 + Vite, :3000) ──HTTP/JSON──► FastAPI (uvicorn, :8000)
                                                   ├─ OceanEmbedInferenceService (singleton)
                                                   │    model + scaler + date index loaded once,
                                                   │    3-month chunk LRU cache
                                                   └─ ArgoValidationService (singleton)
                                                        ArgoCatalog (NetCDF parsed once at startup)
                                                        + per-date full-grid prediction LRU cache
```
Reconstruction: surface tensor `[14,101,241]` → train-only scaler → `OceanEmbedNetV3_Decoder` →
`[15,101,241]` → column at the nearest 0.25° cell → MLD / thermocline depth / OHC300.
ARGO: authentic cast → interpolation onto the 15 depths inside the observed pressure envelope →
same-day, nearest-cell **surface-only** reconstruction → paired metrics on observed depths only.

## 3. Model configuration (verified by execution)

| Item | Verified value |
|:--|:--|
| Class | `OceanEmbedNetV3_Decoder` |
| Trainable parameters | 1,275,934 |
| Checkpoint | `checkpoints/phase5/oceanembed_v3_decoder.pt`, 5,148,823 B, sha256 `29e94f9c…351cf2`, git blob identical to the committed file |
| Input / output | `[14,101,241]` → `[15,101,241]`, all finite |
| Depths | 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m |
| Grid | 5–30°N, 45–105°E, 0.25° |
| Dates | 274 (2020-01-01 → 2020-09-30) |

## 4. Data provenance

* **Inputs:** 7 satellite-derived surface variables (SST, SSS, SSH, U/V current, U/V 10 m wind) + 7 validity masks.
* **Target:** GLORYS12V1 reanalysis regridded to 15 depths. It is a reanalysis reference, **not observational ground truth**.
* **ARGO:** three Coriolis/INCOIS GDAC snapshot files (2020-09-01, -15, -25) in `data/argo/argo2020/`, parsed read-only.

## 5. GLORYS training / evaluation methodology

Chronological split: train Jan–Jul 2020 (213 d), validation Aug 2020 (31 d), held-out test Sep 2020 (30 d).
The scaler means used by the backend were verified at runtime to equal the train-period means in
`configs/scaler_params_experiment_2020_metadata.json` (train window 2020-01-01 → 2020-07-31).
*Offline benchmark* (`reports/phase5/final_model_comparison.json`): v3 GLORYS Sep 2020 RMSE 0.8601 °C,
compared with 1.0418 °C for the Simple CNN, 1.1177 °C for the pointwise MLP, and 2.9976 °C for static climatology.

## 6. ARGO validation methodology (as implemented and tested)

1. The catalog reads every profile. It keeps only profiles inside the domain, with at least 10 physically valid (P, T) pairs, a warmest sample of at least 15 °C, and at least 5 valid target depths.
2. WMO IDs are decoded as characters. This fixes the byte-decoding defect in the older offline ledger.
3. The cast is linearly interpolated onto the 15 depths, with no extrapolation beyond the observed envelope (±5/10 dbar). Unobserved depths remain `null`.
4. Matching: same calendar day (all three dates are in the archive) and nearest 0.25° cell. Offsets: max 10.87 h and mean 5.31 h in time; mean 9.95 km and max 15.72 km in space.
5. The frozen model runs on the **surface tensor only**. `test_10` proves the compared column equals `/predict` at the same cell and date.
6. RMSE, MAE, bias and r are computed only over pairs where both values exist. Per-depth counts sum exactly to 497 (0 m has no observations and reports no statistic).
7. No profile is dropped. Profile `20200901_prof.nc_40` (WMO 6902948) lies on a cell with no valid SST and only 3 of 7 channels. It is kept in the aggregate, as in the published result, and is now flagged explicitly in the API, the aggregate UI and the per-profile UI.

## 7. Backend audit

| Check | Result |
|:--|:--|
| All 8 endpoints | 200 on valid input (localhost and LAN IP) |
| Latitude/longitude out of range | 422 (schema bounds) |
| Land or missing-SST cell | 400 with explanation |
| Unavailable or malformed date | 404 |
| Malformed body | 422 |
| Unknown or malformed ARGO ID (incl. `../`, `%00`) | 404; lookup is a dict key, so no filesystem access |
| ARGO archive missing | 503 on ARGO routes; `/predict` unaffected (tested) |
| Non-finite model output | now rejected instead of served (**fixed**) |
| Stack traces / paths in responses | none (tested) |
| Model / scaler / catalog loading | once per process (tested) |

## 8. Frontend audit

Verified by a Playwright script driving headless Chrome against the real backend (34/34 checks):
* **Reconstruction:**
  * a 15-point profile renders, with MLD, thermocline depth, OHC300 and the surface inputs;
  * a land cell shows an error;
  * changing the location clears the old result;
  * a cleared latitude field disables submit instead of snapping to 5.0.
* **ARGO mode:**
  * the cards show 36 / 28 / 497 / 0.7973 °C / MAE 0.4820 / bias +0.0507, all from `/argo/summary`;
  * the map shows 36 markers;
  * a marker click loads observed and predicted curves, error bars and a 15-row table;
  * selecting another profile changes the metadata, table and metrics, with no stale values;
  * the degraded profile shows its notice.
* Returning to Reconstruction mode still works.
* Browser console: only the deliberate land-cell 400. No uncaught errors and no failed requests.

## 9. Security / configuration audit

* No secrets in the tree. `.env*` is git-ignored; `.env.example` holds only documented optional settings.
* CORS: `*` with `allow_credentials=False`. The API is read-only with no cookies. It can now be restricted through `OCEANEMBED_ALLOWED_ORIGINS`.
* No user-controlled filesystem paths or subprocess calls in the backend. The ARGO profile ID is only a dictionary key, and its echo in errors is sanitised.
* No authentication was added, which is appropriate for a local/LAN research demo.

## 10–11. Tests executed and results

| Command | Result |
|:--|:--|
| `python -m pytest -q` (baseline, before changes) | 30 passed, **1 error** (a script collected as a test) |
| `python -m pytest -q` (final) | **52 passed**, 0 failed |
| `python -m pytest tests/test_argo_api.py -v` | **20 passed** |
| `python -m compileall backend src -q` | exit 0 |
| `npx tsc --noEmit` (`npm run lint`) | exit 0 |
| `npm run build` | success (320 kB JS, 88 kB gzip) |
| HTTP smoke tests (localhost + LAN IP) | all expected status codes (see JSON) |
| Scripted UI demo flow (LAN :3001, localhost :3000) | 34/34 and 34/34 |

New tests added:
* **ARGO:**
  * summary counts, and reproduction of the published RMSE;
  * depth counts summing to 497, and degraded cells being reported;
  * all 36 comparisons, with only observed depths scored;
  * comparison equals the surface-only `/predict`;
  * no stale reuse between profiles;
  * malformed IDs, and a missing archive returning 503.
* **Backend:** malformed dates, schema violations, and finite derived indicators.

## 12. Performance (measured 2026-09-29, CPU)

30 sequential `/predict` requests:
* server total latency: mean **55.3 ms**, median 52.3 ms;
* model forward pass: mean **53.1 ms**;
* HTTP round trip: 62.1 ms;
* outputs identical across all 30 requests.

A warm full ARGO aggregate recompute took 25.6 ms (the first cold computation took about 200 ms). These times are higher than the earlier recorded 35.9 / 31.8 ms. The unchanged forward pass accounts for the whole difference, so it reflects host load at measurement time: two dev servers were running. It is not a regression; the only code added on this path is a finiteness check over 15 values.

## 13–14. Issues found and fixed

| # | Issue | Fix |
|:--|:--|:--|
| 1 | `pytest -q` errored by collecting `scripts/data_pipeline/02_pipeline_test.py` | `pytest.ini` with `testpaths = tests` |
| 2 | `np.trapezoid` requires NumPy ≥ 2; requirements allow 1.24 → `/predict` 500 on NumPy 1.x | Fallback to `np.trapz` (`backend/inference.py`) |
| 3 | Non-finite model outputs could be served as temperatures | Rejected as an invalid cell (400) |
| 4 | Missing ARGO archive gave 500 on `/summary` and `/compare` | 503, matching `/argo/profiles` |
| 5 | Raw user ID (incl. control characters) echoed in the 404 | Sanitised and truncated |
| 6 | Dead code: unused temporal offset and repeated imports in `backend/argo.py` | Single helper `temporal_offset_hours` |
| 7 | "Production-grade" description; root `status: operational` | Research-prototype wording; `status: running` |
| 8 | `ALLOWED_ORIGINS` defined but unused | Wired to CORS; optional env override |
| 9 | ARGO test suite stopped after catalog tests | 14 new ARGO tests (see §10) |
| 10 | Stale reconstruction shown after changing location/date | Prediction cleared on any input change |
| 11 | Cleared coordinate input snapped to 5.0 / 45.0 | Local text state; submit disabled while invalid |
| 12 | Header tooltip hardcoded `127.0.0.1:8000` (wrong on LAN) | Uses the configured `api.baseUrl` |
| 13 | Thermocline band 75–150 m in Reconstruction but 75–200 m in ARGO views | Shared constant `src/lib/ocean.ts` (75–150 m), labelled "nominal" |
| 14 | ARGO map showed a reconstruction crosshair at 15°N 85°E with nothing selected | `showReticle` prop; hidden in ARGO mode |
| 15 | Degraded / non-SST ARGO cell not visible in the aggregate UI | Listed with a link; per-profile "no valid SST" notice |
| 16 | "live comparison", "Quantified uncertainty", "Every claim is checked", unlabelled footer benchmarks, "Operational envelope" | Reworded to on-demand / observed error / offline benchmark / prototype scope |
| 17 | Pipeline cards staggered (9 grid children in 5 columns) | Flex row on desktop, stacked on phone (verified by screenshot) |
| 18 | Favicon 404 in the browser console | Inline SVG favicon |
| 19 | README: "operational", "99 independent floats", Jan–Aug train split, wrong baseline RMSEs, non-existent `/api/*` endpoints, coordinate/day-of-year inputs | Rewritten from verified facts and report files |
| 20 | `src/data/mock.ts` orphan with contradictory metrics (ARGO RMSE 2.191, 8 floats) | Deleted (no importers) |
| 21 | `.env.example` was AI-Studio boilerplate (Gemini key) | Documents `VITE_API_BASE_URL`, `OCEANEMBED_ALLOWED_ORIGINS` |
| 22 | Garbled comment in `vite.config.ts`; launcher advertised port 5173 | Corrected |

## 15. Remaining limitations (not fixed; non-critical)

* The Sep 2020 ARGO set is 36 profiles on 3 days. There is no observational evaluation of other seasons or years.
* ARGO is not strictly independent of GLORYS, which assimilates in-situ data.
* Baseline (CNN/MLP) ARGO and GLORYS figures are offline results. Only v3 is recomputed live.
* Unused npm dependencies (`recharts`, `motion`, `@google/genai`, `express`, `md-to-pdf`, and `cn()` in `src/lib/utils.ts`) were left in place to avoid a lockfile change before the demo.
* `@types/react` is not installed and `tsconfig` is non-strict, so `tsc` gives weak JSX type checking (the IDE reports JSX "any" diagnostics that `tsc` does not).
* Historical planning and phase documents in `docs/` and `reports/` (for example the team decision guide, `PHASE6_BACKEND.md`, and the pre-ARGO `CURRENT_SYSTEM_POSITIONING_AUDIT.md`) keep their original wording as a historical record. `README.md` and this audit are the current, authoritative descriptions.
* Legacy `api/server.py` and `tests/test_e2e.py` (phase-4 path) are still present; the tests pass.
* LAN access was verified via the host's LAN IP from the same machine, not from a second device. Windows Firewall may need to allow ports 3000 and 8000.
* Starlette deprecation warning (`httpx` test client) only.

## 16. SIH demo readiness

**Ready.** Commands:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
npm run dev -- --host 0.0.0.0 --port 3000
# open http://localhost:3000  or  http://<LAN-IP>:3000
```
Suggested flow:
1. **Reconstruction mode:** Bay of Bengal, 2020-09-15, Reconstruct. Show the profile, MLD, thermocline depth and OHC300.
2. **ARGO Validation mode:** show the 36 / 28 / 497 cards and RMSE 0.7973. Click a marker, then a second profile.
3. Scroll to the aggregate depth table: note the thermocline peak at 100 m and that the degraded profile is disclosed.

## 17. Exact verified metrics

| Metric | Value | Source |
|:--|--:|:--|
| ARGO profiles / WMO floats / observations | 36 / 28 / 497 | live `/argo/summary` |
| ARGO RMSE / MAE / Bias / r (v3) | 0.7973 / 0.4820 / +0.0507 / 0.9946 °C | live |
| ARGO RMSE at 100 m (worst depth) | 1.5317 °C | live |
| ARGO RMSE at 500 / 700 / 1000 m | 0.2997 / 0.2631 / 0.2586 °C | live |
| GLORYS Sep 2020 RMSE (v3) | 0.8601 °C | offline benchmark |
| Simple CNN ARGO / GLORYS RMSE | 0.9526 / 1.0418 °C | offline benchmark |

## 18. Unique contribution / differentiation

* One forward pass reconstructs the full 0–1000 m column at 15 standard depths for the whole basin grid, from surface fields alone.
* An interactive, reproducible ARGO check: any judge can pick a float and see observed vs reconstructed profiles, per-depth error, and metrics recomputed by the backend.
* Transparent handling of imperfect data: missing depths are never filled, and degraded input cells are disclosed rather than dropped.
* Strict chronological split with a train-only scaler, verified at runtime.

## 19. Judge-safe claims

* "OceanEmbed reconstructs 15-depth subsurface temperature (0–1000 m) for the North Indian Ocean from seven satellite surface fields."
* "On the held-out September 2020 GLORYS reanalysis, v3 has 0.86 °C RMSE, 17 % lower than our Simple CNN baseline."
* "Against 497 independent in-situ ARGO observations (36 profiles, 28 floats, September 2020), v3 has 0.80 °C RMSE and +0.05 °C bias — recomputed live in the demo."
* "ARGO is used only to evaluate; it is never a model input."
* "Errors are largest in the thermocline (≈1.5 °C at 100 m) and smallest at depth (≈0.26 °C at 1000 m)."
* "Single-profile inference runs in tens of milliseconds on a laptop CPU."

## 20. Claims that must NOT be made

* Real-time, live or operational satellite ingestion, or a live/global ARGO feed.
* Forecasting or prediction of the future. The model reconstructs the state for dates that already have surface data.
* That GLORYS is ground truth, or that ARGO is statistically independent of GLORYS.
* "99 floats", Jan–Aug training, or any baseline number not in the committed reports.
* State-of-the-art or "best", or generalisation beyond Jan–Sep 2020 and the North Indian Ocean.
* PostgreSQL, Redis, JWT, Docker, ONNX/TensorRT, WebGL/3D visualisation, cyclone tracking, or sound-velocity products (none are implemented).
* That per-profile error is an uncertainty estimate.
