"""Entry point: `streamlit run app/main.py`"""
from dotenv import load_dotenv

load_dotenv()

import ui  # noqa: E402  (must load .env before any module reads OLLAMA_URL/OLLAMA_MODEL)

ui.render()
