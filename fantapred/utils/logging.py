import logging, sys
from pathlib import Path

def setup(log_level: str = "INFO", log_file: str | None = None):
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=log_level, format=fmt, handlers=handlers, force=True)

    # Propagate Optuna logs, if Optuna is installed
    try:
        import optuna
        optuna.logging.enable_propagation()
        optuna.logging.set_verbosity(optuna.logging.INFO)
    except ModuleNotFoundError:
        pass
