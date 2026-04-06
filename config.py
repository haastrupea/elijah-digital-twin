# load all the env and extra configs and the swet defaults

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)



_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def get_config() -> dict:

    config = {
        "firstname": "Elijah",
        "name": "Elijah HAASTRUP",
        "openrouter_url": "https://openrouter.ai/api/v1",
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "pushover_user": os.getenv("PUSHOVER_USER"),
        "pushover_token": os.getenv("PUSHOVER_TOKEN"),
        "pushover_url": "https://api.pushover.net/1",
        "project_root": _PROJECT_ROOT
    }

    return config