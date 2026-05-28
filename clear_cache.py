import os, glob

folders = [
    r"C:\Users\Vishu\Desktop\lock-in-jjk\dataset\bw_gege\10_gegeakutami, monochrome manga",
    r"C:\Users\Vishu\Desktop\lock-in-jjk\dataset\color_gege\15_gegecover, painterly watercolor cover",
]

for folder in folders:
    if not os.path.exists(folder):
        print(f"Skipping (not found): {folder}")
        continue
    npz = glob.glob(os.path.join(folder, "*.npz"))
    for f in npz:
        os.remove(f)
    print(f"{folder}: deleted {len(npz)} cache files")

print("Done.")
