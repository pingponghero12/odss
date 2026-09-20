import json
from pathlib import Path

import pytest

import odss


def test_core_scientific_validation_passes_with_numerical_evidence() -> None:
    report = odss.run_scientific_validation()

    assert report.passed
    by_name = {case.name: case for case in report.cases}
    for name in (
        "sgp4_reference_state",
        "frame_and_time",
        "common_epoch",
        "continuous_tca",
        "flux_and_radius_convergence",
        "impact_risk",
        "deterministic_end_to_end",
    ):
        assert by_name[name].status == "pass"
        assert by_name[name].metrics
        assert all(metric.passed for metric in by_name[name].metrics)
    assert set(report.skipped_case_names) == {
        "nasa_sbm_repeatability",
        "propagation_convergence",
        "cascade_vs_brute_force",
    }


def test_validation_report_is_deterministic_and_contains_failure_numbers(tmp_path: Path) -> None:
    failed = odss.ValidationReport(
        (
            odss.ValidationCase(
                "example",
                "fail",
                (odss.ValidationMetric("difference", 2.0, 1.0, 0.1, "m"),),
                "Synthetic report test.",
            ),
        )
    )
    first = odss.validation_report_json(failed)
    second = odss.validation_report_json(failed)
    path = tmp_path / "validation.json"
    odss.write_validation_report(path, failed)
    decoded = json.loads(first)

    assert first == second == path.read_text()
    assert not decoded["passed"]
    metric = decoded["cases"][0]["metrics"][0]
    assert metric["value"] == 2.0
    assert metric["reference"] == 1.0
    assert metric["absolute_error"] == 1.0
    assert metric["absolute_tolerance"] == 0.1


def test_optional_cascade_validation_when_available() -> None:
    pytest.importorskip("cascade")
    pytest.importorskip("heyoka")
    report = odss.run_scientific_validation(include_optional_backends=True)
    by_name = {case.name: case for case in report.cases}

    assert by_name["propagation_convergence"].status == "pass"
    assert by_name["cascade_vs_brute_force"].status == "pass"


def test_optional_nasa_sbm_validation_when_available() -> None:
    pytest.importorskip("nasa_sbm")
    report = odss.run_scientific_validation(include_optional_backends=True)
    by_name = {case.name: case for case in report.cases}

    assert by_name["nasa_sbm_repeatability"].status == "pass"
