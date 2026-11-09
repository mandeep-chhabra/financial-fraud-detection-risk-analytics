"""Dashboard smoke tests.

The full app test needs every artefact (database, model, SHAP files). It is
skipped on a fresh clone so unit tests remain fast and independent.
"""

from __future__ import annotations

import pytest
import streamlit.config as sc
import streamlit.file_util as fu

sc.get_config_files = lambda fn: [fu.get_project_streamlit_file_path(fn)]

from dashboard.common import artifacts_status, money, number, percent
from src.config import DB_PATH, MODEL_PATH, SHAP_DIR

READY = DB_PATH.exists() and MODEL_PATH.exists() and (SHAP_DIR / "shap_values.npy").exists()


def test_formatters():
    assert money(1_500_000) == "$1.5M"
    assert money(2_000) == "$2.0K"
    assert number(1234567) == "1,234,567"
    assert percent(12.3456) == "12.35%"


def test_artifacts_status_reports_paths():
    status = artifacts_status()
    assert set(status) == {
        "database", "model", "encoder", "thresholds", "predictions",
        "shap", "quality_report", "test_metrics",
    }
    assert all(isinstance(v, bool) for v in status.values())


@pytest.mark.skipif(not READY, reason="dashboard artefacts not built")
def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=180)
    app.run()
    assert not app.exception
    # The sidebar radio lists all six pages.
    assert len(app.radio) == 1
    assert len(app.radio[0].options) == 6
    assert app.title or len(app.metric) > 0
