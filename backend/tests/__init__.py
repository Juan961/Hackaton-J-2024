import os
import json
import logging


try:
    with open('./local.settings.json') as f:
        data = json.load(f)

        env = data["Values"]

        for k, v in env.items():
            os.environ[k] = v

except FileNotFoundError:
    logging.error("Error loading local settings file")
