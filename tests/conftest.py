"""Keep a developer's suite file (argus_cortex.settings) out of the tests."""

import os

os.environ["ARGUS_ENV_FILE"] = "off"
