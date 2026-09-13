"""
SIH26066 — OceanEmbed Root Application Entry Point
Enables running:
    uvicorn main:app --reload
    python main.py
or:
    uvicorn backend.main:app --reload
from the project root directory.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
