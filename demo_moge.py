#!/usr/bin/env python3
"""
MoGe demo script that outputs in megasam_extracted format.

Output structure:
- rgb/: RGB images as PNG files (00000.png, 00001.png, ...)
- depth/: Depth images as 16-bit PNG files (00000.png, 00001.png, ...)
- cam_K.txt: Average camera intrinsics matrix (3x3)

No mask folder is generated.
"""

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
import torch
import tqdm
from moge.model.v2 import MoGeModel

LONG_DIM = 640


def demo(model, args):
    outdir = args.outdir
    scene_name = args.scene_name

    # Create output directory structure
    outdir_scene = os.path.join(outdir, scene_name)
    rgb_dir = os.path.join(outdir_scene, "rgb")
    depth_dir = os.path.join(outdir_scene, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # Get image list
    img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
    img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))
    img_path_list = sorted(img_path_list)

    if len(img_path_list) == 0:
        print(f"No images found in {args.img_path}")
        return

    print(f"Processing {len(img_path_list)} images...")

    # Store all intrinsics for averaging
    all_intrinsics = []

    for idx, img_path in enumerate(tqdm.tqdm(img_path_list)):
        # Read RGB image
        rgb = np.array(Image.open(img_path))[..., :3]
        original_h, original_w = rgb.shape[:2]

        # Resize to LONG_DIM while maintaining aspect ratio
        if rgb.shape[1] > rgb.shape[0]:
            final_w, final_h = LONG_DIM, int(
                round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
            )
        else:
            final_w, final_h = (
                int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
                LONG_DIM,
            )

        rgb_resized = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)

        # Prepare input for MoGe
        rgb_torch = torch.from_numpy(rgb_resized / 255).permute(2, 0, 1)

        # Predict
        predictions = model.infer(rgb_torch)

        # Extract intrinsics
        intri = predictions["intrinsics"].clone()
        intri[0, 0] = intri[0, 0] * final_w  # fx
        intri[1, 1] = intri[1, 1] * final_h  # fy
        intri[0, 2] = intri[0, 2] * final_w  # cx
        intri[1, 2] = intri[1, 2] * final_h  # cy

        # Extract depth
        depth = predictions["depth"].cpu().numpy()  # (H, W)

        # Resize depth back to original resolution if needed
        if depth.shape[0] != original_h or depth.shape[1] != original_w:
            depth = cv2.resize(depth, (original_w, original_h), cv2.INTER_LINEAR)

        # Scale intrinsics to original resolution
        scale_x = original_w / final_w
        scale_y = original_h / final_h
        intri_original = intri.cpu().numpy().copy()
        intri_original[0, 0] *= scale_x  # fx
        intri_original[1, 1] *= scale_y  # fy
        intri_original[0, 2] *= scale_x  # cx
        intri_original[1, 2] *= scale_y  # cy

        all_intrinsics.append(intri_original)

        # Save RGB image
        rgb_output_path = os.path.join(rgb_dir, f"{idx:05d}.png")
        cv2.imwrite(rgb_output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Convert depth to uint16 (scale to millimeters)
        # Assuming depth is in meters, convert to millimeters
        depth_mm = depth * 1000.0
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        # Save depth image
        depth_output_path = os.path.join(depth_dir, f"{idx:05d}.png")
        cv2.imwrite(depth_output_path, depth_uint16)

    # Compute average intrinsics
    avg_intrinsics = np.mean(all_intrinsics, axis=0)

    # Save camera intrinsics
    cam_K_path = os.path.join(outdir_scene, "cam_K.txt")
    np.savetxt(cam_K_path, avg_intrinsics, fmt='%.8f')

    print(f"\nProcessing complete!")
    print(f"Output saved to: {outdir_scene}")
    print(f"- RGB images: {len(img_path_list)} files in rgb/")
    print(f"- Depth images: {len(img_path_list)} files in depth/")
    print(f"- Camera intrinsics: cam_K.txt")
    print(f"\nAverage camera intrinsics:")
    print(avg_intrinsics)


def main():
    parser = argparse.ArgumentParser(
        description='MoGe demo that outputs in megasam_extracted format'
    )
    parser.add_argument("--img-path", type=str, required=True,
                        help="Path to input images directory")
    parser.add_argument("--outdir", type=str, default="./moge_output",
                        help="Output directory")
    parser.add_argument("--scene-name", type=str, required=True,
                        help="Scene name for output subdirectory")

    args = parser.parse_args()

    print("Torch version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from huggingface hub
    print("Loading MoGe model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    print("Model loaded successfully!")

    demo(model, args)


if __name__ == "__main__":
    main()
