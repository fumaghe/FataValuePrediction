from joblib import Memory
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
memory = Memory(CACHE_DIR, verbose=0)
