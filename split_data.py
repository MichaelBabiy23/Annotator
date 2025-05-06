#!/usr/bin/env python3
import os
import random
import shutil
import argparse

def split_data_move(data_dir: str, val_ratio: float, seed: int = None):
    # 1) Locate your “Train” folders under images/ and labels/
    images_train = os.path.join(data_dir, 'images', 'Train')
    labels_train = os.path.join(data_dir, 'labels', 'Train')
    if not os.path.isdir(images_train):
        raise FileNotFoundError(f"Could not find train-images at {images_train}")
    if not os.path.isdir(labels_train):
        raise FileNotFoundError(f"Could not find train-labels at {labels_train}")

    # 2) Choose your validation target folders.
    #    Prefer data/images/val & data/labels/val if they exist;
    #    otherwise fall back to data/validation/{images,labels}
    images_val = os.path.join(data_dir, 'images', 'val')
    labels_val = os.path.join(data_dir, 'labels', 'val')
    if not os.path.isdir(images_val):
        images_val = os.path.join(data_dir, 'validation', 'images')
    if not os.path.isdir(labels_val):
        labels_val = os.path.join(data_dir, 'validation', 'labels')

    # 3) Create them if needed
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # 4) List all your train images
    all_imgs = [
        fn for fn in os.listdir(images_train)
        if fn.lower().endswith(('.jpg','jpeg','png'))
    ]
    print(f"🔍 Found {len(all_imgs)} train images in {images_train!r}")

    # 5) Compute how many go to validation
    n_val = int(len(all_imgs) * val_ratio)
    if n_val == 0 and all_imgs:
        n_val = 1
        print("⚠️  Ratio too small – forcing at least 1 into validation")

    if seed is not None:
        random.seed(seed)

    # 6) Pick them at random
    val_imgs = random.sample(all_imgs, n_val)
    print(f"✂️  Moving {n_val} images → {images_val!r}")

    # 7) Move each image + its label
    for img in val_imgs:
        # move image
        src_i = os.path.join(images_train, img)
        dst_i = os.path.join(images_val, img)
        print(f"  • {img}")
        shutil.move(src_i, dst_i)

        # move corresponding label
        lbl = os.path.splitext(img)[0] + '.txt'
        src_l = os.path.join(labels_train, lbl)
        dst_l = os.path.join(labels_val, lbl)
        if os.path.exists(src_l):
            shutil.move(src_l, dst_l)
        else:
            print(f"    ⚠️  no label for {img!r} (expected {lbl})")

    print("\n✅ Done.")
    print("Images now in:", images_val)
    print("Labels now in:", labels_val)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Move a random subset from images/Train + labels/Train into your val folders"
    )
    p.add_argument(
        "-d", "--data-dir",
        default="data",
        help="Root data folder (contains images/Train, labels/Train, etc.)"
    )
    p.add_argument(
        "-v", "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of Train → validation (e.g. 0.2 = 20%)"
    )
    p.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    args = p.parse_args()
    split_data_move(args.data_dir, args.val_ratio, args.seed)
