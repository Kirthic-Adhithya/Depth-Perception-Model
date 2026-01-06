import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np


class KITTIDataset(Dataset):
    def __init__(self, root_dir, pseudo_dir=None, transform=None):
        self.transform = transform
        self.image_files = []
        self.depth_files = []

        # --- Real KITTI data ---
        self.img_dir = os.path.join(root_dir, "train", "image")
        self.depth_dir = os.path.join(root_dir, "train", "depth")

        if os.path.exists(self.img_dir) and os.path.exists(self.depth_dir):
            img_files = sorted(glob(os.path.join(self.img_dir, "*.*")))
            dep_files = sorted(glob(os.path.join(self.depth_dir, "*.*")))

            img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
            dep_names = [os.path.splitext(os.path.basename(f))[0] for f in dep_files]
            common = list(set(img_names) & set(dep_names))

            img_final = [
                os.path.join(self.img_dir, n + ".jpg") if os.path.exists(os.path.join(self.img_dir, n + ".jpg"))
                else os.path.join(self.img_dir, n + ".png")
                for n in common
            ]
            dep_final = [
                os.path.join(self.depth_dir, n + ".png")
                for n in common if os.path.exists(os.path.join(self.depth_dir, n + ".png"))
            ]

            self.image_files.extend(img_final)
            self.depth_files.extend(dep_final)
            print(f"[INFO] Found {len(common)} real KITTI pairs.")
        else:
            print("[WARN] No real KITTI data found!")

        # --- Pseudo data ---
        if pseudo_dir:
            pseudo_img_dir = os.path.join(pseudo_dir, "unlabeled_images")
            pseudo_depth_dir = os.path.join(pseudo_dir, "pseudo_depths")

            if os.path.exists(pseudo_img_dir) and os.path.exists(pseudo_depth_dir):
                img_files = sorted(glob(os.path.join(pseudo_img_dir, "*.*")))
                dep_files = sorted(glob(os.path.join(pseudo_depth_dir, "*.*")))

                img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
                dep_names = [os.path.splitext(os.path.basename(f))[0] for f in dep_files]
                common = list(set(img_names) & set(dep_names))

                img_final = []
                dep_final = []
                for n in common:
                    img_path_jpg = os.path.join(pseudo_img_dir, n + ".jpg")
                    img_path_png = os.path.join(pseudo_img_dir, n + ".png")
                    dep_path_png = os.path.join(pseudo_depth_dir, n + ".png")

                    if os.path.exists(dep_path_png):
                        if os.path.exists(img_path_jpg):
                            img_final.append(img_path_jpg)
                            dep_final.append(dep_path_png)
                        elif os.path.exists(img_path_png):
                            img_final.append(img_path_png)
                            dep_final.append(dep_path_png)

                self.image_files.extend(img_final)
                self.depth_files.extend(dep_final)
                print(f"[INFO] Found {len(img_final)} pseudo-labeled pairs.")
            else:
                print("[WARN] Pseudo directories not found!")

        print(f"[INFO] Using {len(self.image_files)} image–depth pairs total.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")

        # Convert depth safely to grayscale
        depth = Image.open(self.depth_files[idx])
        if depth.mode != "L":
            depth = depth.convert("L")

        if self.transform:
            img = self.transform(img)
            depth = self.transform(depth)

        return img, depth


def get_data_loaders(root_dir, pseudo_dir=None, batch_size=2):
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor()
    ])

    dataset = KITTIDataset(root_dir, pseudo_dir=pseudo_dir, transform=transform)

    val_split = int(0.8 * len(dataset))
    train_data = list(range(val_split))
    val_data = list(range(val_split, len(dataset)))

    train_subset = torch.utils.data.Subset(dataset, train_data)
    val_subset = torch.utils.data.Subset(dataset, val_data)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Using {len(train_subset)} image–depth pairs for training.")
    print(f"[INFO] Using {len(val_subset)} image–depth pairs for validation.")

    return train_loader, val_loader
