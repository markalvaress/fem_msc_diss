# run with `pytest tests/test_all.py` (from root directory)

import os
import subprocess

def test_all():
    # Check that all experiments run without error.
    # Fails at the first failed experiment
    experiments = os.listdir("experiments")
    for exp in experiments:
        result = subprocess.run(["bash", f"experiments/{exp}"], capture_output = True)
        if result.returncode != 0:
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            print("experiment " + exp + " failed. Stdout: '" + stdout + "', stderr:'" + stderr + "'.")
            assert(False)

    assert(True)