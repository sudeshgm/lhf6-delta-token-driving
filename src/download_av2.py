"""Download AV2 Sensor ring_front_center at 2 Hz (every 10th frame of 20 Hz) + city_SE3_egovehicle.feather.
Frames are resized (shorter side 512) to JPEG q90 to save disk. Usage: python src/download_av2.py [max_logs]"""
import subprocess, sys, os, glob, cv2
S5 = os.environ.get("S5CMD", "/tmp/s5cmd"); B = "s3://argoverse/datasets/av2/sensor"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "av2")
def ls(p):
    r = subprocess.run([S5, "--no-sign-request", "ls", p], capture_output=True, text=True).stdout.split("\n")
    return [l.split()[-1] for l in r if l.strip()]
def main(maxlogs=10**9):
    logs = [("val", l.rstrip("/")) for l in ls(f"{B}/val/")] + [("train", l.rstrip("/")) for l in ls(f"{B}/train/")]
    for sp, log in logs[:maxlogs]:
        d = f"{OUT}/{sp}/{log}"; cam = f"{d}/ring_front_center"
        if os.path.exists(f"{d}/DONE"): continue
        os.makedirs(cam, exist_ok=True)
        imgs = sorted(ls(f"{B}/{sp}/{log}/sensors/cameras/ring_front_center/"))[::10]
        cmds = [f"cp {B}/{sp}/{log}/city_SE3_egovehicle.feather {d}/"] + [f"cp {B}/{sp}/{log}/sensors/cameras/ring_front_center/{f} {cam}/{f}" for f in imgs]
        subprocess.run([S5, "--no-sign-request", "--numworkers", "32", "run"], input="\n".join(cmds), text=True, capture_output=True)
        for f in glob.glob(f"{cam}/*.jpg"):
            im = cv2.imread(f); h, w = im.shape[:2]; s = 512 / min(h, w)
            cv2.imwrite(f, cv2.resize(im, (round(w*s), round(h*s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
        open(f"{d}/DONE", "w").close(); print(sp, log, len(imgs), flush=True)
if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 10**9)
