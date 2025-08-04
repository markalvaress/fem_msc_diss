from datetime import datetime 
import os

sim_outputs_folder = "./sim_outputs"

def dt_now():
    now = str(datetime.now().replace(second=0, microsecond=0))[:-3].replace(" ", "_").replace(":", "-")
    return now

def init_outfolder(sim_name: str) -> str:
    """Creates the full output folder path, and creates the folders in the filesystem if they do not already exist."""
    out_folder = f"{sim_outputs_folder}/{sim_name}"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    return out_folder

def done(out_folder: str) -> None:
    print("Simulation complete.")
    print("Output saved to: " + out_folder)
    return
    