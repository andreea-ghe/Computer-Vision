import os
import torch
import hashlib
import torchvision
from PIL import Image
import numpy as np


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, transforms=None, download=False, split_name="train"):
        super().__init__()
        self.transforms = transforms
        self.base_folder = base_folder

        self.images_dir = os.path.join(base_folder, "lfw_funneled")
        self.masks_dir = os.path.join(base_folder, "parts_lfw_funneled_gt_images")

        print("Images dir:", self.images_dir)
        print("Exists?", os.path.exists(self.images_dir))

        print("Masks dir:", self.masks_dir)
        print("Exists?", os.path.exists(self.masks_dir))

        self.samples = []

        # ---------------------------
        # 🔥 STEP 1: build mask lookup
        # ---------------------------
        mask_lookup = {}

        for root, _, files in os.walk(self.masks_dir):
            for mask_name in files:
                if not mask_name.endswith(".ppm"):
                    continue

                clean_name = mask_name

                # remove macOS prefix
                if clean_name.startswith("._"):
                    clean_name = clean_name[2:]

                key = clean_name.replace(".ppm", "").lower().strip()
                mask_lookup[key] = os.path.join(root, mask_name)

        print("Total masks found:", len(mask_lookup))
        print("Example mask keys:", list(mask_lookup.keys())[:5])

        # ---------------------------
        # 🔥 STEP 2: match images
        # ---------------------------
        for person in os.listdir(self.images_dir):
            img_person_path = os.path.join(self.images_dir, person)

            if not os.path.isdir(img_person_path):
                continue

            for img_name in os.listdir(img_person_path):
                if img_name.startswith("._"):
                    continue

                img_path = os.path.join(img_person_path, img_name)

                key = img_name.replace(".jpg", "").lower().strip()

                if key in mask_lookup:
                    self.samples.append((img_path, mask_lookup[key]))

        print("TOTAL MATCHED SAMPLES:", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError("No matching image-mask pairs found")

        # ---------------------------
        # 🔥 STEP 3: split
        # ---------------------------
        self.samples.sort(key=lambda x: hashlib.md5(x[0].encode()).hexdigest())

        split_idx = int(0.8 * len(self.samples))
        if split_name == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    # ---------------------------
    # 🔥 GET ITEM
    # ---------------------------
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        # ⚠️ IMPORTANT: try opening mask
        try:
            mask = Image.open(mask_path)
        except Exception as e:
            raise RuntimeError(f"Mask failed to open: {mask_path}") from e

        if self.transforms:
            image, mask = self.transforms(image, mask)
        else:
            image = torchvision.transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

    def __len__(self):
        return len(self.samples)


# ---------------------------
# 🔥 TEST
# ---------------------------
if __name__ == "__main__":
    dataset = LFWDataset(base_folder="lab4/lfw_dataset", transforms=None)

    print("Dataset size:", len(dataset))

    img, mask = dataset[0]

    print("Image shape:", img.shape)  # [3, H, W]
    print("Mask shape:", mask.shape)  # [H, W]
    print("Mask dtype:", mask.dtype)
    print("Mask values:", torch.unique(mask))