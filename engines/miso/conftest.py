"""Keep a developer's suite file out of the tests: miso_server.py copies it into the environment at import
(argus_cortex.settings.apply), and a failed assert could print a value from it."""

import os

os.environ["ARGUS_ENV_FILE"] = "off"
