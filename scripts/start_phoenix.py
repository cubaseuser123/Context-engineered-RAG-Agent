import phoenix as px
import time
import os
from pathlib import Path

os.environ.setdefault("PHOENIX_WORKING_DIR", str(Path.home() / ".phoenix"))
px.launch_app(use_temp_dir=False)

print("Phoenix is running on http://localhost:6006")
while True:
    time.sleep(10)
