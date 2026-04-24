import argparse
import glob
import os

import cv2
import imageio
import numpy as np
from PIL import Image
import torch
import tqdm
from moge.model.v2 import MoGeModel # Let's try MoGe-2

LONG_DIM = 640

def demo(model, args):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)
  # img_path_list = sorted(glob.glob("/home/zhengqili/filestore/DAVIS/DAVIS/JPEGImages/480p/%s/*.jpg"%scene_name))
  img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))

  fovs = []
  for img_path in tqdm.tqdm(img_path_list):
    rgb = np.array(Image.open(img_path))[..., :3]
    if rgb.shape[1] > rgb.shape[0]:
      final_w, final_h = LONG_DIM, int(
          round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
      )
    else:
      final_w, final_h = (
          int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
          LONG_DIM,
      )
    rgb = cv2.resize(
        rgb, (final_w, final_h), cv2.INTER_AREA
    )  # .transpose(2, 0, 1)

    rgb_torch = torch.from_numpy(rgb / 255).permute(2, 0, 1)
    # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    # predict
    predictions = model.infer(rgb_torch)

    intri = predictions["intrinsics"].clone()
    intri[0, 0] = intri[0, 0] * final_w  # fx
    intri[1, 1] = intri[1, 1] * final_h  # fy
    intri[0, 2] = intri[0, 2] * final_w  # fx
    intri[1, 2] = intri[1, 2] * final_h  # fy

    fov_ = np.rad2deg(
        2
        * np.arctan(
            predictions["depth"].shape[-1]
            / (2 * intri[0, 0].cpu().numpy())
        )
    )
    depth = predictions["depth"].cpu().numpy()
    print(fov_)
    fovs.append(fov_)
    # breakpoint()
    np.savez(
        os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
        depth=np.float32(depth),
        fov=fov_,
    )


from huggingface_hub import hf_hub_download
import json

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--img-path", type=str)
  parser.add_argument("--outdir", type=str, default="./vis_depth")
  parser.add_argument("--scene-name", type=str)

  args = parser.parse_args()

  print("Torch version:", torch.__version__)

  device = torch.device("cuda")

  # Load the model from huggingface hub (or load from local).
  model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device) 

  demo(model, args)