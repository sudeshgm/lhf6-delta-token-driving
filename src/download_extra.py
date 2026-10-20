"""Fetch calibration (intrinsics, egovehicle_SE3_sensor) + annotations.feather for every downloaded log."""
import glob, os, subprocess
S5 = os.environ.get("S5CMD", "/tmp/s5cmd"); B = "s3://argoverse/datasets/av2/sensor"; R = os.path.join(os.path.dirname(__file__), "..")
cmds = []
for d in sorted(os.path.dirname(p) for p in glob.glob(f"{R}/data/av2/*/*/DONE")):
    sp, log = d.split("/")[-2:]
    for f in ["calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather", "annotations.feather"]:
        if not os.path.exists(f"{d}/{os.path.basename(f)}"): cmds.append(f"cp {B}/{sp}/{log}/{f} {d}/{os.path.basename(f)}")
subprocess.run([S5, "--no-sign-request", "--numworkers", "64", "run"], input="\n".join(cmds), text=True); print(len(cmds))
