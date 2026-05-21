# Executa o pipeline completo ao rodar o script diretamente
import os
import sys
import numpy as np
import random
from PIL import Image
from utils.terminal_to_pdf import TerminalToPDF
from io_cityscapes import list_images, list_labelids, get_resolutions

# Import analysis/plotting functions
from analysis_cityscapes import plot_pixel_frequency_per_class, plot_class_appearance_per_image, print_class_frequencies

# Import label definitions dynamically
import importlib

LABELS_MODULE = "helpers.cityscapes_labels"  # Labels module 
labels_mod = importlib.import_module(LABELS_MODULE)
id2label = labels_mod.id2label
labels = labels_mod.labels

# === CONFIGURATION FOR DATASET CITYSCAPES ===
IMAGE_SUFFIX = "_leftImg8bit.png"  # RGB images
MASK_SUFFIX = "_gtFine_labelIds.png"  # Label masks
IMG_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
MASK_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/gtFine_trainvaltest/gtFine"

def analyze_classes_and_frequencies(labelids_list):
    '''Analyzes class frequencies and appearances in the dataset'''
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

    # Prepare class names dict for plotting
    class_names = {cls_id: id2label[cls_id].name if cls_id in id2label else str(cls_id) for cls_id in pixel_counts}

    # Plot pixel frequency per class
    plot_pixel_frequency_per_class(pixel_counts, class_names, save_path="pixel_frequency_per_class.png")
    print("Gráfico de frequência de pixels por classe salvo como pixel_frequency_per_class.png")


    # Plot class appearance per image
    plot_class_appearance_per_image(class_appears_in, class_names, save_path="class_appearance_per_image.png")
    print("Gráfico de aparição das classes por imagem salvo como class_appearance_per_image.png")

    # Print detailed class analysis (textual)
    print_class_frequencies(
        pixel_counts, total_pixels, class_appears_in, labelids_list,
        id2label, MASK_SUFFIX, IMAGE_SUFFIX, MASK_DIR, IMG_DIR
    )


if __name__ == "__main__":
    buffer = TerminalToPDF("terminal_output.pdf")
    buffer.start()
    try:
        images = list_images(IMG_DIR, IMAGE_SUFFIX)
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
        labelids = list_labelids(MASK_DIR, MASK_SUFFIX)
        print(f"\nTotal labelIds masks found: {len(labelids)}")

        # Analyze classes and pixel frequencies in labelIds masks
        analyze_classes_and_frequencies(labelids)
    finally:
        buffer.stop()
        buffer.save_pdf()
