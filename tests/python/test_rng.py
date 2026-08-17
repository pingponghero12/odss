import pytest

import odss


def test_random_values_match_the_published_philox_vector() -> None:
    key = odss.RandomKey(master_seed=0, scenario_id=0, run_id=0, object_id=0, stream_id=0)

    assert [odss.random_u64(key, index) for index in range(4)] == [
        0x16554D9ECA36314C,
        0xDB20FE9D672D0FDC,
        0xD7E772CEE186176B,
        0x7E68B68AEC7BA23B,
    ]


def test_random_key_is_immutable_and_supports_random_access() -> None:
    key = odss.RandomKey(master_seed=1, scenario_id=2, run_id=3, object_id=4, stream_id=5)
    expected = [odss.random_u64(key, index) for index in (100, 3, 99, 3)]

    assert expected[1] == expected[3]
    assert key == odss.RandomKey(1, 2, 3, 4, 5)
    with pytest.raises(AttributeError):
        key.run_id = 8


def test_named_streams_have_stable_independent_ids() -> None:
    assert odss.named_stream_id("fragment_mass") == 0x90579DF240ACAA5C
    assert odss.named_stream_id("fragment_mass") == odss.named_stream_id("fragment_mass")
    assert odss.named_stream_id("fragment_mass") != odss.named_stream_id("fragment_velocity")

    mass_key = odss.named_random_key(
        master_seed=42,
        scenario_id=1,
        run_id=2,
        object_id=3,
        stream_name="fragment_mass",
    )
    velocity_key = odss.named_random_key(
        master_seed=42,
        scenario_id=1,
        run_id=2,
        object_id=3,
        stream_name="fragment_velocity",
    )
    assert mass_key.stream_id == odss.named_stream_id("fragment_mass")
    assert odss.random_u64(mass_key, 0) != odss.random_u64(velocity_key, 0)


def test_named_random_key_validates_its_inputs() -> None:
    with pytest.raises(ValueError):
        odss.named_stream_id("  ")
    with pytest.raises(ValueError):
        odss.named_random_key(
            master_seed=-1,
            scenario_id=0,
            run_id=0,
            object_id=0,
            stream_name="sample",
        )
    with pytest.raises(ValueError):
        odss.named_random_key(
            master_seed=True,
            scenario_id=0,
            run_id=0,
            object_id=0,
            stream_name="sample",
        )


def test_uniform_values_are_in_the_half_open_unit_interval() -> None:
    key = odss.named_random_key(
        master_seed=19,
        scenario_id=8,
        run_id=7,
        object_id=6,
        stream_name="uniform_test",
    )

    values = [odss.uniform_01(key, index) for index in range(100)]
    assert all(0.0 <= value < 1.0 for value in values)
    assert values == [odss.uniform_01(key, index) for index in range(100)]
