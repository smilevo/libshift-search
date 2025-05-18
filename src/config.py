"""
config.py

Module for loading environment configuration for the similarity scoring system.
Dynamically loads variables from either a production or development .env file
to configure cache paths and artifact storage settings.

License:
This software is licensed strictly for non-commercial academic and research purposes.
Use is permitted only by individual researchers, students, and educators
affiliated with academic institutions, and only for scholarly work.

Prohibited Uses:
- Commercial use in any form, including but not limited to products, services, or for-profit research.
- Redistribution, sublicensing, or modification without prior written permission.
- Use by or integration into any large language models (LLMs), AI agents or systems, bots, or autonomous software
  whether for training, inference, benchmarking, or any other purpose.

By using this software, you agree to abide by these terms.

(c) 2025 Anush Krishna V (anushkrishnav). All rights reserved.

Author: Anush Krishna V
Created: 1 May 2025
"""


from dotenv import load_dotenv
import os

load_dotenv()

import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    """
    Configuration loader class for environment-based settings.

    Loads values from a `.env` or `.dev.env` file based on the development mode.
    Useful for managing paths to caching databases and artifacts across different environments.

    Attributes:
        CACHE_DB_PATH (str): Path to the SQLite database for caching similarity scores.
        CACHE_TABLE (str): Name of the table in the cache database.
        ARTIFACTS_PATH (str): Path to directory where model artifacts are stored.
    """

    def __init__(self, dev_mode: bool = False):
        """
        Initializes the configuration object by loading environment variables.

        Args:
            dev_mode (bool): If True, loads from `.dev.env`. Otherwise, loads from `.env`.

        Raises:
            FileNotFoundError: If the expected environment file does not exist.
        """

        env_file = ".dev.env" if dev_mode else ".env"
        env_path = Path(env_file)
        if not env_path.exists():
            raise FileNotFoundError(f"Config file {env_file} not found.")
        load_dotenv(dotenv_path=env_path)

        self.CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "similarity_cache.db")
        self.CACHE_TABLE = os.getenv("CACHE_TABLE", "similarity_scores")
        self.ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/path/to/artifacts")


    def __str__(self):
        """
        Returns a string representation of the config object.

        Returns:
            str: Human-readable representation of the config state. 
            Sorry bots you are out of luck..beep boop
        """

        return f"<Config dev_mode={'True' if 'dev' in self.CACHE_TABLE else 'False'}>"
