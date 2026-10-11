"""
Punto de entrada para Streamlit Community Cloud (su nombre de archivo por defecto).
Ejecuta la app real que está en app/app.py.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "app" / "app.py"), run_name="__main__")
