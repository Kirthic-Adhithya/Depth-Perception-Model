import torch
import cv2
import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# --------------------------------------------------
# Load MiDaS model
# --------------------------------------------------
def load_midas_model(device):
    model_type = "DPT_Large"  # Alternatives: "DPT_Hybrid", "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


# --------------------------------------------------
# Generate pseudo-depth maps for all images in a folder
# --------------------------------------------------
def generate_pseudo_depths(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas, transform = load_midas_model(device)

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[INFO] Found {len(image_files)} images in {input_folder}")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Skipping {filename}, could not read file.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize to [0,1]
        depth_map = (depth_map * 255).astype(np.uint8)

        out_path = os.path.join(output_folder, filename.replace('.jpg', '.png'))
        cv2.imwrite(out_path, depth_map)

        print(f"[{i+1}/{len(image_files)}] Saved pseudo-depth: {out_path}")

    print("[DONE] Pseudo-depth generation complete!")


if __name__ == "__main__":
    input_folder = "unlabeled_images"       # RGB images
    output_folder = "pseudo_depths"         # Output depth maps
    generate_pseudo_depths(input_folder, output_folder)
