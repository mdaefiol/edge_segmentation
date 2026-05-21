from utils.terminal_to_pdf import TerminalToPDF
import os
import sys
import numpy as np
from PIL import Image

# Import local cityscapes label definitions (generic for future datasets)
from helpers.cityscapes_labels import id2label, labels

# Path to cityscapes images
CITYSCAPES_IMG_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
# Path to labelIds masks
CITYSCAPES_LABELIDS_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/gtFine_trainvaltest/gtFine"


# List all images in train/val/test splits
def list_images(base_path):
    images = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(
            base_path, split
        )  # Path to the current split (train, val, or test)

        # Check if the directory exists
        if not os.path.isdir(split_dir):
            continue

        # Iterate over the cities within the split
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)

            # Check if the city directory exists
            if not os.path.isdir(city_dir):
                continue

            # Iterate over the files within the city directory
            for filename in os.listdir(city_dir):
                if filename.endswith("_leftImg8bit.png"):
                    images.append(os.path.join(city_dir, filename))

    # Return list of image paths found
    return images


# Get image resolutions
def get_resolutions(image_list):
    resolutions = []  # List to store resolutions

    for img_path in image_list:
        try:
            with Image.open(img_path) as img:  # Open the image with PIL
                resolutions.append(img.size)  # (width, height)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    # Return list of resolutions
    return resolutions


# List labelIds masks in train/val/test splits
def list_labelids(base_path):
    masks = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_path, split)
        if not os.path.isdir(split_dir):
            continue
        for city in os.listdir(split_dir):
            city_dir = os.path.join(
                split_dir, city
            )  # Path to the city directory within the split
            if not os.path.isdir(city_dir):
                continue
            for filename in os.listdir(city_dir):
                if filename.endswith("_gtFine_labelIds.png"):
                    # File ending with _gtFine_labelIds.png is a labelIds mask
                    masks.append(os.path.join(city_dir, filename))
    return masks


# Analyze classes and pixel frequencies
def analyze_classes_and_frequencies(labelids_list):
    pixel_counts = {}  # Dictionary to count pixels per class (cls_id: count)
    total_pixels = 0
    
    # For frequency of appearance of each class per image
    class_appears_in = {}  # cls_id: number of images in which it appears
    
    for mask_path in labelids_list:
        try:
            mask = np.array(Image.open(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            
            # Update pixel count
            for cls_id, count in zip(unique, counts):
                pixel_counts[cls_id] = pixel_counts.get(cls_id, 0) + count
                # Count occurrence
                class_appears_in[cls_id] = class_appears_in.get(cls_id, 0) + 1
            total_pixels += mask.size
        
        except Exception as e:
            print(f"Error opening mask {mask_path}: {e}")

    print_class_frequencies(pixel_counts, total_pixels, class_appears_in, labelids_list)


def print_class_frequencies(pixel_counts, total_pixels, class_appears_in, labelids_list):
    print("\nAbsolute and relative pixel frequency per class:")
    print("Class           ID   Pixels      Relative (%)")
    print("-" * 45)

    for cls_id, count in sorted(pixel_counts.items(), key=lambda x: -x[1]):
        label = id2label.get(cls_id)
        name = label.name if label else str(cls_id)
        rel = 100 * count / total_pixels if total_pixels else 0
        print(f"{name:15} {cls_id:3} {count:10} {rel:12.2f}")
    print(f"\nTotal pixels: {total_pixels}")

    # Build list of present classes (with label name)
    present = []
    for cls_id in pixel_counts:
        if cls_id in id2label:
            present.append(id2label[cls_id].name)
    print(f"Classes present ({len(present)}): {present}")

    # Frequency of appearance of each class per image
    print("\nFrequency of appearance of each class per image:")
    num_imgs = len(labelids_list)
    
    for cls_id, freq in sorted(class_appears_in.items(), key=lambda x: -x[1]):
        label = id2label.get(cls_id)
        name = label.name if label else str(cls_id)
        print(
            f"{name:<20} {cls_id:<5} {freq:<8} ({100*freq/num_imgs:.2f}% of images)"
        )

    # Degree of class imbalance (ratio between the most and least frequent)
    if len(pixel_counts) > 1:
        # Find class with max and min pixel count (excluding zero)
        max_cls_id, max_count = max(pixel_counts.items(), key=lambda x: x[1])
        
        # Filter out classes with zero pixels for min calculation
        nonzero_pixel_counts = []
        for item in pixel_counts.items():
            if item[1] > 0:
                nonzero_pixel_counts.append(item)
        min_cls_id, min_count = min(nonzero_pixel_counts, key=lambda x: x[1])
        max_label = id2label.get(max_cls_id)
        min_label = id2label.get(min_cls_id)
        
        if min_count > 0:
            imbalance = max_count / min_count
            print(f"\nDegree of class imbalance (max/min): {imbalance:.2f}")
            print(f"Most frequent class: {max_label.name if max_label else max_cls_id} (ID {max_cls_id}) with {max_count} pixels")
            print(f"Least frequent class: {min_label.name if min_label else min_cls_id} (ID {min_cls_id}) with {min_count} pixels")
        else:
            print("\nDegree of class imbalance: infinite (min=0)")
    else:
        print("\nDegree of class imbalance: not applicable (only one class)")


if __name__ == "__main__":
    buffer = TerminalToPDF("terminal_output.pdf")
    buffer.start()
    try:
        images = list_images(CITYSCAPES_IMG_DIR)
        print(f"Total images found: {len(images)}")

        # Pass the list of images to get the resolutions
        resolutions = get_resolutions(images)
        if resolutions:
            width, height = resolutions[0]
            print(f"Typical resolution: {width}x{height} (first image)")

            # Check if all have the same resolution
            unique_resolutions = set(resolutions)  # set = get unique elements
            print(f"Unique resolutions found: {unique_resolutions}")
        else:
            print("No resolutions found.")

        # Analyze classes and pixel frequencies
        labelids = list_labelids(
            CITYSCAPES_LABELIDS_DIR
        )  # List of labelIds mask paths
        print(f"\nTotal labelIds masks found: {len(labelids)}")

        # Analyze classes and pixel frequencies in labelIds masks
        analyze_classes_and_frequencies(labelids)
    finally:
        buffer.stop()
        buffer.save_pdf()
