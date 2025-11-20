import logging
import logging.config
import os
import yaml
from pathlib import Path

def setup_logging(
    default_path: str = None,
    default_level: int = logging.INFO,
    env_key: str = "LOG_CFG"
) -> None:
    """
    Setup logging configuration from YAML file.
    If the environment variable LOG_CFG is set, it uses that path.
    Otherwise uses default_path.
    """
    path = default_path or os.getenv(env_key, None)
    if path and os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.getLogger().warning(
            f"Logging configuration file not found at {path}; using basicConfig with level {default_level}"
        )

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance under the 'fraud_detection' namespace.
    """
    return logging.getLogger(f"fraud_detection.{name}")

# Optionally, you can call setup_logging at import time:
# For example, when your application starts (in run_api.py or pipeline_runner.py),
# you do:
#
# from fraud_detection.logging.logger import setup_logging
# import fraud_detection.configs.app as cfg
# setup_logging(default_path=cfg.logging.config_path, default_level=cfg.logging.log_level)
#
# Then you call get_logger everywhere:
# logger = get_logger(__file__)
