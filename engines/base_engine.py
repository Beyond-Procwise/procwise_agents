# ProcWise/engines/base_engine.py

import os
import json

# Define the project's root directory dynamically
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class BaseEngine:
    """A base class for all engines, providing common utilities like file loading."""

    def _load_json_file(self, relative_path: str) -> dict:
        """Loads a JSON file using an absolute path constructed from the project root."""
        absolute_path = os.path.join(PROJECT_ROOT, relative_path)
        try:
            with open(absolute_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR ({self.__class__.__name__}): Could not find file at: {absolute_path}")
            raise
        except Exception as e:
            print(
                f"FATAL ERROR ({self.__class__.__name__}): Failed to load/parse JSON from {absolute_path}. Error: {e}")
            raise
