import os
import shutil
import importlib
import sys
import pytest


@pytest.fixture()
def test_env(tmp_path, monkeypatch):
    """Isolate data location and return imported modules (cli, controller, core, model).
    Sets MWI_DATA_LOCATION to a temporary directory before importing project modules.
    """
    # Point app to an isolated temp data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MWI_DATA_LOCATION", str(data_dir))

    # Ensure a clean import state for modules that depend on settings
    for name in list(sys.modules.keys()):
        if name in ("settings", "mwi.model", "mwi.core", "mwi.controller", "mwi.cli"):
            sys.modules.pop(name, None)

    # Import modules after env var is set so settings picks it up
    from mwi import cli as _cli
    from mwi import controller as _controller
    from mwi import model as _model
    from mwi import core as _core
    # Force settings.data_location to the temp dir for any path-based logic
    import settings as _settings
    _settings.data_location = str(data_dir)
    # Also ensure controller module references the updated settings value
    _controller.settings.data_location = str(data_dir)

    # Return modules and the data dir for convenience
    return {
        "cli": _cli,
        "controller": _controller,
        "model": _model,
        "core": _core,
        "data_dir": data_dir,
    }


@pytest.fixture()
def fresh_db(test_env, monkeypatch):
    """Create/drop tables for a clean DB using DbController.setup with auto-confirm.
    Returns the same dict as test_env.
    """
    core = test_env["core"]
    controller = test_env["controller"]
    data_dir = test_env["data_dir"]
    # Ensure DB file exists and (re)init peewee DB to this path
    db_path = os.path.join(str(data_dir), "mwi.db")
    if not os.path.exists(db_path):
        open(db_path, "a").close()
    model = test_env["model"]
    # Rebind DB in case it captured an older path
    model.DB.init(db_path, pragmas={
        'journal_mode': 'wal',
        'cache_size': -1 * 512000,
        'foreign_keys': 1,
        'ignore_check_constrains': 0,
        'synchronous': 0
    })
    try:
        model.DB.connect(reuse_if_open=True)
    except Exception:
        # Connection may be opened lazily later
        pass
    # Auto-confirm destructive actions (patch both module refs)
    monkeypatch.setattr(core, "confirm", lambda msg: True, raising=True)
    monkeypatch.setattr(controller.core, "confirm", lambda msg: True, raising=True)
    # Setup database (drop + create tables)
    ret = controller.DbController.setup(core.Namespace())
    assert ret == 1
    return test_env
