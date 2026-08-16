import math

import pytest

import odss


def epoch() -> odss.Epoch:
    return odss.Epoch(offset_s=123.5, reference_epoch="J2000", time_scale="TAI")


def frame() -> odss.ReferenceFrame:
    return odss.ReferenceFrame(identifier="GCRF")


def population() -> odss.ParticlePopulation:
    return odss.ParticlePopulation(
        epoch=epoch(),
        frame=frame(),
        position_x_m=[1.0, 2.0],
        position_y_m=[3.0, 4.0],
        position_z_m=[5.0, 6.0],
        velocity_x_m_s=[7.0, 8.0],
        velocity_y_m_s=[9.0, 10.0],
        velocity_z_m_s=[11.0, 12.0],
        mass_kg=[100.0, 200.0],
        area_m2=[2.0, 4.0],
    )


def test_epoch_and_reference_frame_are_explicit_immutable_values() -> None:
    value = epoch()
    reference_frame = frame()

    assert value.offset_s == 123.5
    assert value.reference_epoch == "J2000"
    assert value.time_scale == "TAI"
    assert value == epoch()
    assert reference_frame.identifier == "GCRF"
    assert reference_frame == frame()
    with pytest.raises(AttributeError):
        value.offset_s = 0.0
    with pytest.raises(AttributeError):
        reference_frame.identifier = "ITRF"


@pytest.mark.parametrize(
    ("offset_s", "reference_epoch", "time_scale"),
    [
        (math.nan, "J2000", "TAI"),
        (0.0, "", "TAI"),
        (0.0, "J2000", "  "),
    ],
)
def test_epoch_rejects_invalid_metadata(
    offset_s: float, reference_epoch: str, time_scale: str
) -> None:
    with pytest.raises(ValueError):
        odss.Epoch(offset_s, reference_epoch, time_scale)


def test_cartesian_state_has_si_values_frame_and_epoch() -> None:
    state = odss.CartesianState(
        position_m=(1.0, 2.0, 3.0),
        velocity_m_s=(4.0, 5.0, 6.0),
        epoch=epoch(),
        frame=frame(),
    )

    assert state.position_m == (1.0, 2.0, 3.0)
    assert state.velocity_m_s == (4.0, 5.0, 6.0)
    assert state.epoch == epoch()
    assert state.frame == frame()
    assert state == odss.CartesianState(
        (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), epoch(), frame()
    )
    with pytest.raises(ValueError):
        odss.CartesianState((math.inf, 0.0, 0.0), (0.0, 0.0, 0.0), epoch(), frame())
    with pytest.raises(AttributeError):
        state.position_m = (0.0, 0.0, 0.0)


def test_physical_properties_are_positive_finite_si_values() -> None:
    properties = odss.PhysicalProperties(mass_kg=10.0, area_m2=2.5)

    assert properties.mass_kg == 10.0
    assert properties.area_m2 == 2.5
    assert properties == odss.PhysicalProperties(10.0, 2.5)
    with pytest.raises(ValueError):
        odss.PhysicalProperties(0.0, 1.0)
    with pytest.raises(ValueError):
        odss.PhysicalProperties(1.0, math.nan)


def test_particle_population_exposes_immutable_soa_fields() -> None:
    particles = population()

    assert len(particles) == 2
    assert particles.size == 2
    assert not particles.empty
    assert particles.epoch == epoch()
    assert particles.frame == frame()
    assert particles.position_x_m == (1.0, 2.0)
    assert particles.position_y_m == (3.0, 4.0)
    assert particles.position_z_m == (5.0, 6.0)
    assert particles.velocity_x_m_s == (7.0, 8.0)
    assert particles.velocity_y_m_s == (9.0, 10.0)
    assert particles.velocity_z_m_s == (11.0, 12.0)
    assert particles.mass_kg == (100.0, 200.0)
    assert particles.area_m2 == (2.0, 4.0)
    assert particles == population()
    with pytest.raises(TypeError):
        particles.position_x_m[0] = 99.0
    with pytest.raises(AttributeError):
        particles.mass_kg = (1.0, 2.0)


def test_particle_population_allows_consistent_empty_fields() -> None:
    particles = odss.ParticlePopulation(epoch(), frame(), [], [], [], [], [], [], [], [])

    assert particles.empty
    assert len(particles) == 0


def test_particle_population_rejects_inconsistent_or_invalid_fields() -> None:
    valid_fields = ([1.0],) * 8

    with pytest.raises(ValueError):
        odss.ParticlePopulation(epoch(), frame(), [1.0], [], *valid_fields[2:])
    with pytest.raises(ValueError):
        odss.ParticlePopulation(
            epoch(), frame(), [math.nan], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]
        )
    with pytest.raises(ValueError):
        odss.ParticlePopulation(
            epoch(), frame(), [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [-1.0], [1.0]
        )


def test_minimal_experiment_backend_and_observable_specs() -> None:
    backend = odss.BackendSpec(kind="reference")
    observable = odss.ObservableSpec(kind="state_count")
    experiment = odss.ExperimentSpec(
        population=population(), backend=backend, observables=(observable,)
    )

    assert backend.kind == "reference"
    assert observable.kind == "state_count"
    assert experiment.population == population()
    assert experiment.backend == backend
    assert experiment.observables == (observable,)
    assert experiment == odss.ExperimentSpec(population(), backend, [observable])
    with pytest.raises(AttributeError):
        experiment.backend = odss.BackendSpec("other")
    with pytest.raises(ValueError):
        odss.BackendSpec("")
    with pytest.raises(ValueError):
        odss.ObservableSpec("\n")
