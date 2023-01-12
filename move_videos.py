"""
Temporary script to reorganise all of the output files of my simple app into a 
single directory, with each file being named after its creation datetime, and a 
single meta file with all the metadata for each file.
"""
from pathlib import Path
import json
import shutil


IN_DIR = Path("out")
OUT_DIR = Path("collated_out")

OUT_DIR.mkdir(exist_ok=True)

metadata = {}

# Each trial_dir should contain a single motion-detected video, config.json, stats.json,
# and output.log
for trial_dir in IN_DIR.iterdir():
    if not trial_dir.is_dir():
        # print(f"ERROR: Not a directory: {fp}")
        continue

    stats_fp = trial_dir / "stats.json"
    if not stats_fp.exists():
        # print(f"ERROR: No stats: {fp}")
        continue

    with open(stats_fp, "r") as f:
        stats = json.load(f)
        # print(f"Loaded stats for {fp}")

    config_fp = trial_dir / "config.json"
    if not config_fp.exists():
        # print(f"ERROR: No config: {fp}")
        continue

    with open(config_fp, "r") as f:
        config = json.load(f)
        # print(f"Loaded config for {fp}")

    # Finally copy the actual video file to where it needs to be
    video_fp = trial_dir / "motion001.avi"
    start_datetime = trial_dir.name
    video_copy_fp = (OUT_DIR / start_datetime).with_suffix(video_fp.suffix)

    print(f"Copying {trial_dir} --> {video_copy_fp}")
    shutil.copy(src=video_fp, dst=video_copy_fp)

    metadata[video_copy_fp.name] = {
        "config": config,
        "stats": stats
    }

# Now save metadata to file in that same directory? Or in a different directory?
# print(metadata)
metadata_fp = OUT_DIR / "meta.json"
with open(metadata_fp, "w") as f:
    json.dump(metadata, f)