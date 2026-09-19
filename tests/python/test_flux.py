import math

import pytest

import odss


def population(count: int, *, masses_kg: list[float] | None = None) -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=odss.Epoch(0.0, "2026-06-20T12:00:00", "TAI"),
        frame=odss.ReferenceFrame("GCRS"),
        position_x_m=[7_000_000.0] * count,
        position_y_m=[0.0] * count,
        position_z_m=[0.0] * count,
        velocity_x_m_s=[0.0] * count,
        velocity_y_m_s=[7_500.0] * count,
        velocity_z_m_s=[0.0] * count,
        mass_kg=masses_kg or [1.0] * count,
        area_m2=[1.0] * count,
    )


def test_analytical_beam_number_mass_and_energy_flux() -> None:
    debris = population(2, masses_kg=[2.0, 3.0])
    targets = population(1)
    events = (
        odss.ConjunctionEvent(0, 0, 2.0, 5.0, 10.0),
        odss.ConjunctionEvent(1, 0, 8.0, 6.0, 20.0),
    )
    spec = odss.FluxSpec(10.0, (0.0, 10.0), (0.0, math.inf))

    result = odss.evaluate_flux(events, debris, targets, (0.01, 0.02), spec)

    assert result.spec == spec
    assert result.target_count == 1
    assert len(result.bins) == 1
    value = result.bins[0]
    denominator = math.pi * 10.0**2 * 10.0
    assert value.encounter_count == 2
    assert value.number_flux_m2_s == pytest.approx(2.0 / denominator)
    assert value.mass_flux_kg_m2_s == pytest.approx(5.0 / denominator)
    assert value.kinetic_energy_flux_w_m2 == pytest.approx(
        (0.5 * 2.0 * 10.0**2 + 0.5 * 3.0 * 20.0**2) / denominator
    )


def test_target_time_and_size_bins_are_explicit_and_complete() -> None:
    debris = population(3, masses_kg=[1.0, 2.0, 3.0])
    targets = population(2)
    events = (
        odss.ConjunctionEvent(0, 0, 2.0, 1.0, 10.0),
        odss.ConjunctionEvent(1, 1, 7.0, 1.0, 10.0),
        odss.ConjunctionEvent(2, 1, 7.0, 20.0, 10.0),
    )
    spec = odss.FluxSpec(5.0, (0.0, 5.0, 10.0), (0.0, 0.1, math.inf))

    result = odss.evaluate_flux(events, debris, targets, (0.05, 0.2, 0.3), spec)

    assert len(result.bins) == 8
    nonzero = [value for value in result.bins if value.encounter_count]
    assert [(value.target_index, value.time_start_s, value.size_min_m) for value in nonzero] == [
        (0, 0.0, 0.0),
        (1, 5.0, 0.1),
    ]


def test_sampling_radius_convergence_for_area_scaled_synthetic_encounters() -> None:
    debris = population(9)
    targets = population(1)
    distances_m = (0.5,) + (1.5,) * 3 + (2.5,) * 5
    events = tuple(
        odss.ConjunctionEvent(index, 0, 5.0, distance_m, 1.0)
        for index, distance_m in enumerate(distances_m)
    )

    results = odss.evaluate_flux_radius_convergence(
        events,
        debris,
        targets,
        (0.1,) * 9,
        sampling_radii_m=(1.0, 2.0, 3.0),
        time_bin_edges_s=(0.0, 10.0),
        size_bin_edges_m=(0.0, math.inf),
    )

    assert [result.bins[0].encounter_count for result in results] == [1, 4, 9]
    assert [result.bins[0].number_flux_m2_s for result in results] == pytest.approx(
        [1.0 / (10.0 * math.pi)] * 3
    )


def test_empty_events_produce_zero_target_wise_bins() -> None:
    result = odss.evaluate_flux(
        (),
        population(0),
        population(2),
        (),
        odss.FluxSpec(1_000.0, (0.0, 60.0), (0.0, math.inf)),
    )

    assert len(result.bins) == 2
    assert all(value.encounter_count == 0 for value in result.bins)
    assert all(value.number_flux_m2_s == 0.0 for value in result.bins)


@pytest.mark.parametrize(
    "spec",
    [
        lambda: odss.FluxSpec(0.0, (0.0, 1.0), (0.0, math.inf)),
        lambda: odss.FluxSpec(1.0, (1.0, 2.0), (0.0, math.inf)),
        lambda: odss.FluxSpec(1.0, (0.0, math.inf), (0.0, math.inf)),
        lambda: odss.FluxSpec(1.0, (0.0, 1.0), (0.0, 1.0, 0.5)),
    ],
)
def test_flux_spec_rejects_invalid_geometry(spec: object) -> None:
    with pytest.raises(ValueError):
        spec()


def test_flux_rejects_mismatched_physical_metadata_and_indices() -> None:
    debris = population(1)
    targets = population(1)
    spec = odss.FluxSpec(10.0, (0.0, 10.0), (0.0, math.inf))

    with pytest.raises(ValueError, match="match debris"):
        odss.evaluate_flux((), debris, targets, (), spec)
    with pytest.raises(ValueError, match="outside"):
        odss.evaluate_flux(
            (odss.ConjunctionEvent(2, 0, 1.0, 1.0, 1.0),),
            debris,
            targets,
            (0.1,),
            spec,
        )
