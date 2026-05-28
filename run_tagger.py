import sys
import os

# Hardcoded paths — avoids cmd comma-in-path splitting issue
FOLDER = r"C:\Users\Vishu\Desktop\lock-in-jjk\dataset\bw_gege\10_gegeakutami, monochrome manga"
TAGGER = r"C:\Users\Vishu\Desktop\lock-in-jjk\training\kohya_ss\sd-scripts\finetune\tag_images_by_wd14_tagger.py"

sys.argv = [
    TAGGER,
    "--repo_id", "SmilingWolf/wd-vit-tagger-v3",
    "--thresh", "0.35",
    "--caption_extension", ".txt",
    "--remove_underscore",
    "--append_tags",
    "--onnx",
    FOLDER,
]

os.chdir(r"C:\Users\Vishu\Desktop\lock-in-jjk")

with open(TAGGER, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, TAGGER, "exec"))
