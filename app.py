"""Root entry-point shim for Streamlit Cloud (`streamlit run app.py`).

The real page moved to ``backend/streamlit/main.py`` during the package
reorg. Streamlit RE-EXECUTES the entry script on every rerun (widget
interaction), so this uses ``runpy.run_path`` rather than a plain import:
an import would be cached in ``sys.modules`` after the first run and the
page would freeze on every subsequent rerun. ``run_path`` re-executes the
page module fresh each time, exactly as Streamlit expects.
"""
import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "backend", "streamlit", "main.py"),
    run_name="__main__",
)
