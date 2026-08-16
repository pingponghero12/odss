def test_import() -> None:
    import odss

    assert odss is not None


def test_version() -> None:
    import odss

    assert isinstance(odss.version(), str)
    assert odss.version()
