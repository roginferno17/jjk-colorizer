import os, glob
folder = r"C:\Users\Vishu\Desktop\lock-in-jjk\dataset\bw_gege\10_gegeakutami, monochrome manga"
npz = glob.glob(os.path.join(folder, "*.npz"))
print(f"Found {len(npz)} stale .npz latent files")
for f in npz:
    os.remove(f)
print("Done — cache cleared.")
