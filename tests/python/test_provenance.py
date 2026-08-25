from dataclasses import FrozenInstanceError

import pytest

import odss


def population(*, first_position_x_m: float = 1.0) -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=odss.Epoch(offset_s=123.5, reference_epoch="J2000", time_scale="TAI"),
        frame=odss.ReferenceFrame(identifier="GCRF"),
        position_x_m=[first_position_x_m, 2.0],
        position_y_m=[3.0, 4.0],
        position_z_m=[5.0, 6.0],
        velocity_x_m_s=[7.0, 8.0],
        velocity_y_m_s=[9.0, 10.0],
        velocity_z_m_s=[11.0, 12.0],
        mass_kg=[100.0, 200.0],
        area_m2=[2.0, 4.0],
    )


def experiment(
    *,
    first_position_x_m: float = 1.0,
    observables: tuple[str, ...] = ("count", "summary"),
) -> odss.ExperimentSpec:
    return odss.ExperimentSpec(
        population=population(first_position_x_m=first_position_x_m),
        backend=odss.BackendSpec(kind="reference"),
        observables=tuple(odss.ObservableSpec(kind=item) for item in observables),
    )


def test_equivalent_experiments_have_identical_canonical_identity() -> None:
    first = experiment()
    equivalent = experiment(observables=("summary", "count"))

    assert first is not equivalent
    assert odss.canonical_experiment(first) == odss.canonical_experiment(equivalent)
    assert odss.experiment_hash(first) == odss.experiment_hash(equivalent)


def test_experiment_identity_has_a_known_answer() -> None:
    canonical = odss.canonical_experiment(experiment())

    assert canonical.startswith('{"backend":{"kind":"reference"}')
    assert '"offset_s":"0x1.ee00000000000p+6"' in canonical
    assert " " not in canonical
    assert (
        odss.experiment_hash(experiment())
        == "911436d8eb0697873be4fc32075b6fcdb9223415f176cc9ed699da2e9433e6f1"
    )


def test_experiment_identity_changes_with_scientific_content() -> None:
    assert odss.experiment_hash(experiment()) != odss.experiment_hash(
        experiment(first_position_x_m=1.5)
    )


def test_equivalent_positive_and_negative_zero_have_one_identity() -> None:
    assert odss.experiment_hash(experiment(first_position_x_m=0.0)) == odss.experiment_hash(
        experiment(first_position_x_m=-0.0)
    )


def test_study_identity_is_independent_of_experiment_order() -> None:
    first = experiment()
    second = experiment(first_position_x_m=1.5)

    assert odss.canonical_study((first, second)) == odss.canonical_study((second, first))
    assert odss.study_hash((first, second)) == odss.study_hash((second, first))
    assert odss.study_hash((first,)) != odss.study_hash((first, second))


def test_identity_functions_reject_unsupported_inputs() -> None:
    with pytest.raises(TypeError):
        odss.canonical_experiment("not an experiment")
    with pytest.raises(TypeError):
        odss.canonical_study("not a study")
    with pytest.raises(TypeError):
        odss.canonical_study((experiment(), "not an experiment"))


def test_input_asset_metadata_is_validated_normalized_and_immutable() -> None:
    asset = odss.InputAssetMetadata(
        logical_name="catalog.oml",
        content_sha256="AB" * 32,
        size_bytes=42,
    )

    assert asset.content_sha256 == "ab" * 32
    with pytest.raises(FrozenInstanceError):
        asset.size_bytes = 43
    with pytest.raises(ValueError):
        odss.InputAssetMetadata("", "ab" * 32, 1)
    with pytest.raises(ValueError):
        odss.InputAssetMetadata("catalog.oml", "not-a-digest", 1)
    with pytest.raises(ValueError):
        odss.InputAssetMetadata("catalog.oml", "ab" * 32, -1)


def test_software_metadata_is_validated_and_reports_odss_version() -> None:
    metadata = odss.odss_software_metadata()

    assert metadata == odss.SoftwareMetadata(name="odss", version=odss.version())
    with pytest.raises(ValueError):
        odss.SoftwareMetadata(name="", version="1.0")


def test_run_manifest_is_immutable_and_canonical() -> None:
    experiment_id = odss.experiment_hash(experiment())
    study_id = odss.study_hash((experiment(),))
    first_asset = odss.InputAssetMetadata("first", "01" * 32, 1)
    second_asset = odss.InputAssetMetadata("second", "02" * 32, 2)
    odss_metadata = odss.odss_software_metadata()
    dependency = odss.SoftwareMetadata("dependency", "2.0")
    manifest = odss.RunManifest(
        experiment_hash=experiment_id.upper(),
        master_seed=123456,
        scenario_id=7,
        run_id=9,
        study_hash=study_id.upper(),
        input_assets=[second_asset, first_asset],
        software=[odss_metadata, dependency],
    )
    reordered = odss.RunManifest(
        experiment_hash=experiment_id,
        master_seed=123456,
        scenario_id=7,
        run_id=9,
        study_hash=study_id,
        input_assets=(first_asset, second_asset),
        software=(dependency, odss_metadata),
    )

    assert manifest.experiment_hash == experiment_id
    assert manifest.study_hash == study_id
    assert manifest.master_seed == 123456
    assert manifest.scenario_id == 7
    assert manifest.run_id == 9
    assert manifest.rng_algorithm == odss.RNG_ALGORITHM
    assert isinstance(manifest.input_assets, tuple)
    assert isinstance(manifest.software, tuple)
    canonical = odss.canonical_manifest(manifest)
    assert canonical == odss.canonical_manifest(reordered)
    assert canonical.startswith('{"experiment_hash":')
    assert '"master_seed":123456' in canonical
    assert '"rng_algorithm":"philox4x64-10-v1"' in canonical
    with pytest.raises(FrozenInstanceError):
        manifest.study_hash = None


def test_run_manifest_rejects_invalid_provenance() -> None:
    experiment_id = odss.experiment_hash(experiment())

    with pytest.raises(ValueError):
        odss.RunManifest(experiment_hash="invalid", master_seed=1, scenario_id=2, run_id=3)
    with pytest.raises(TypeError):
        odss.RunManifest(
            experiment_hash=experiment_id,
            master_seed=1,
            scenario_id=2,
            run_id=3,
            input_assets=("invalid",),
        )
    with pytest.raises(TypeError):
        odss.RunManifest(
            experiment_hash=experiment_id,
            master_seed=1,
            scenario_id=2,
            run_id=3,
            software=("invalid",),
        )
    for field_name in ("master_seed", "scenario_id", "run_id"):
        values = {"master_seed": 1, "scenario_id": 2, "run_id": 3}
        values[field_name] = -1
        with pytest.raises(ValueError):
            odss.RunManifest(experiment_hash=experiment_id, **values)
    with pytest.raises(ValueError):
        odss.RunManifest(
            experiment_hash=experiment_id,
            master_seed=1,
            scenario_id=2,
            run_id=3,
            rng_algorithm="",
        )
