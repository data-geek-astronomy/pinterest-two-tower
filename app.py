# Entry point for Hugging Face Spaces
# HF Spaces looks for app.py at the repo root by default.
# We simply import and re-export the Streamlit app from app/streamlit_app.py.

import runpy
import os
from pathlib import Path

# Make sure imports inside streamlit_app.py resolve correctly
os.chdir(Path(__file__).parent)

runpy.run_path("app/streamlit_app.py", run_name="__main__")
