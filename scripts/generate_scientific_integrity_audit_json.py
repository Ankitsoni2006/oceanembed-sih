import os
import json

audit_data = {
    "audit_title": "SIH26066 Pre-Phase-5 Scientific Integrity and Claims Audit",
    "timestamp": "2026-09-13T15:35:00Z",
    "auditor": "Antigravity Scientific Quality Gate Engine",
    "status": "AUDIT COMPLETE — CLAIMS RECONCILED",
    "issues_evaluated": {
        "issue_1_generalization_gap": {
            "claim": "near-zero generalization gap (|1.8142 - 1.8126| = 0.0016 C)",
            "verdict": "REMOVE",
            "scientific_rationale": (
                "Generalization gap strictly denotes the delta between training and test loss/error "
                "on identically distributed samples. GLORYS is a continuous 730,230-cell reanalysis grid, "
                "while ARGO is a sparse set of 497 point CTD measurements across 3 snapshot dates. "
                "Their sampling distributions, vertical geometries, and representativeness differ fundamentally. "
                "Calling 0.0016 C a 'generalization gap' is scientifically invalid and a coincidence of aggregate RMSE."
            ),
            "recommended_wording": "similar aggregate RMSE magnitude (~1.81 C) across the two evaluation references"
        },
        "issue_2_consistent_generalization_and_calibration": {
            "claim": "Consistent Generalization / The model is well-calibrated",
            "verdict": "REMOVE",
            "scientific_rationale": (
                "Simple CNN is the best-performing model on both GLORYS (1.0418 C) and ARGO-2020 (0.9526 C). "
                "Pointwise MLP is second (1.1177 C on GLORYS, 1.1602 C on ARGO). OceanEmbed is third on both "
                "(1.8142 C and 1.8126 C). Describing OceanEmbed as exhibiting 'consistent generalization' obscures "
                "that it is underperforming simpler baselines by ~0.8 C on both references. Furthermore, "
                "'well-calibrated' is a technical term requiring uncertainty quantification / calibration curve analysis, "
                "which was not performed."
            ),
            "recommended_wording": (
                "The ARGO-2020 evaluation reproduces a similar error scale (~1.81 C) to the GLORYS test, but the current "
                "OceanEmbed architecture underperforms the simpler CNN and MLP baselines on both evaluation references."
            )
        },
        "issue_3_september_2020_argo_coverage": {
            "claim": "September 2020 ARGO validation (unqualified)",
            "verdict": "SOFTEN",
            "scientific_rationale": (
                "The acquired ARGO dataset is not continuous across all 30 days of September 2020. "
                "It consists strictly of 3 snapshot dates: September 1 (13 profiles, 180 points), "
                "September 15 (15 profiles, 207 points), and September 25 (8 profiles, 110 points), "
                "totaling 36 authentic profiles and 497 points."
            ),
            "recommended_wording": (
                "Contemporaneous ARGO evaluation using 36 vertical profiles available across three snapshot dates "
                "in September 2020 (September 1, 15, and 25)."
            )
        },
        "issue_4_independent_argo_terminology": {
            "claim": "independent observational validation (without qualification)",
            "verdict": "SOFTEN",
            "scientific_rationale": (
                "ARGO is completely independent of OceanEmbed's training input stream (satellite surface fields). "
                "However, GLORYS12V1 is a data-assimilating ocean reanalysis that assimilates in-situ observations. "
                "Whether these exact 36 ARGO profiles were assimilated into GLORYS12V1 was not independently verified."
            ),
            "recommended_wording": (
                "ARGO provides an in-situ observational evaluation that is independent of the model's training inputs. "
                "Assimilation relationship between these exact ARGO profiles and the GLORYS reference was not verified."
            )
        },
        "issue_5_overstated_claims_count": 6,
        "issue_6_definitive_model_comparison": {
            "cnn_currently_best": True,
            "mlp_currently_second": True,
            "oceanembed_currently_third": True,
            "table": {
                "climatology": {"glorys_test_rmse": 2.9976, "argo_2020_rmse": 3.0363},
                "pointwise_mlp": {"glorys_test_rmse": 1.1177, "argo_2020_rmse": 1.1602},
                "simple_cnn": {"glorys_test_rmse": 1.0418, "argo_2020_rmse": 0.9526},
                "oceanembed": {"glorys_test_rmse": 1.8142, "argo_2020_rmse": 1.8126}
            }
        },
        "issue_7_thermocline_weakness": {
            "thermocline_is_major_weakness": True,
            "observed_error_range": "2.01 C at 75m, 3.25 C at 100m, 3.14 C at 125m, 2.49 C at 150m",
            "scientific_framing": (
                "OceanEmbed's largest ARGO errors occur in the approximately 75–150 m depth range, indicating that "
                "thermocline-region reconstruction is a major current performance limitation."
            )
        },
        "issue_8_model_improvement_justified": {
            "justified": True,
            "rationale": (
                "Evidence clearly shows OceanEmbed underperforming Simple CNN (1.81 C vs 0.95 C) across both GLORYS and ARGO, "
                "with error heavily localized in the 75–150m thermocline. Improving the model through controlled experiments "
                "(e.g., thermocline-weighted loss or capacity tuning) is justified as a development goal."
            )
        },
        "issue_9_data_leakage_and_isolation": {
            "status": "PASS",
            "argo_in_training": False,
            "argo_in_scaler": False,
            "argo_in_checkpoint_selection": False,
            "argo_in_hyperparameter_tuning": False,
            "splits_preserved": "Train: Jan-Jul (213d), Val: Aug (31d), Test: Sep (30d)",
            "checkpoints_frozen": True
        }
    },
    "verdict_summary": {
        "issue_1_near_zero_generalization_gap": "REMOVE",
        "issue_2_consistent_generalization": "REMOVE",
        "issue_3_september_2020_coverage": "SOFTEN",
        "issue_4_independent_argo_terminology": "SOFTEN",
        "issue_5_overstated_claims_count": 6,
        "issue_6_cnn_currently_best": "YES",
        "issue_7_thermocline_is_major_weakness": "YES",
        "issue_8_model_improvement_justified": "YES",
        "issue_9_data_leakage": "PASS",
        "experiment_modified": "NO",
        "phase_4_artifacts_preserved": "YES",
        "scientifically_defensible_for_sih": "YES",
        "phase_5_ready": "NO (Model improvement / controlled refinement should be undertaken before final dashboard presentation)"
    }
}

os.makedirs("reports/argo2020", exist_ok=True)
with open("reports/argo2020/scientific_integrity_audit.json", "w") as f:
    json.dump(audit_data, f, indent=2)

print("Saved reports/argo2020/scientific_integrity_audit.json successfully.")
