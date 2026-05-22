# Executa o pipeline completo ao rodar o script diretamente

import os
import sys
import numpy as np
import random
from PIL import Image
import json
import csv
import matplotlib.pyplot as plt

from utils.terminal_to_pdf import TerminalToPDF
from io_cityscapes import list_images, list_labelids, get_resolutions
from report_cityscapes import generate_cityscapes_report


# Import analysis/plotting functions
from analysis_cityscapes import plot_pixel_frequency_per_class, plot_class_appearance_per_image, print_class_frequencies, generate_all_overlays

# Import dataset configuration
from dataset_config import IMAGE_SUFFIX, MASK_SUFFIX, IMG_DIR, MASK_DIR, LABELS_MODULE

# Import label definitions dynamically
import importlib
labels_mod = importlib.import_module(LABELS_MODULE)
id2label = labels_mod.id2label
labels = labels_mod.labels


def plot_resolution_histogram(resolutions, save_path="resolution_histogram.png"):
    """Plots and saves a histogram of image resolutions (width x height)."""
    if not resolutions:
        print("No resolutions to plot.")
        return
    widths, heights = zip(*resolutions)
    plt.figure(figsize=(10, 5))
    plt.hist(widths, bins=20, alpha=0.7, label='Width')
    plt.hist(heights, bins=20, alpha=0.7, label='Height')
    plt.xlabel('Pixels')
    plt.ylabel('Number of Images')
    plt.title('Histogram of Image Resolutions (Width and Height)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Resolution histogram saved as {save_path}")

def export_statistics_csv_json(pixel_counts, total_pixels, class_appears_in, resolutions, csv_path="stats_summary.csv", json_path="stats_summary.json"):
    """Exports statistics to CSV and JSON files."""
    # CSV: class_id, class_name, pixel_count, pixel_percent, appears_in_images
    os.makedirs("outputs/statistics", exist_ok=True)
    
    csv_path = "outputs/statistics/stats_summary.csv"
    json_path = "outputs/statistics/stats_summary.json"
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class_id", "class_name", "pixel_count", "pixel_percent", "appears_in_images"])
        
        for cls_id, count in pixel_counts.items():
            name = id2label[cls_id].name if cls_id in id2label else str(cls_id)
            percent = 100 * count / total_pixels if total_pixels else 0
            appears = class_appears_in.get(cls_id, 0)
            writer.writerow([cls_id, name, count, percent, appears])
    print(f"Statistics exported to {csv_path}")

    # JSON: summary dict
    summary = {
        "classes": [
            {
                "class_id": int(cls_id),
                "class_name": id2label[cls_id].name if cls_id in id2label else str(cls_id),
                "pixel_count": int(pixel_counts[cls_id]),
                "pixel_percent": 100 * pixel_counts[cls_id] / total_pixels if total_pixels else 0,
                "appears_in_images": int(class_appears_in.get(cls_id, 0))
            }
            for cls_id in pixel_counts
        ],
        "total_pixels": int(total_pixels),
        "resolutions": [
            {"width": int(w), "height": int(h)} for (w, h) in resolutions
        ]
    }
    
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print(f"Statistics exported to {json_path}")

def analyze_classes_and_frequencies(labelids_list, resolutions):
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
    plot_pixel_frequency_per_class(pixel_counts, class_names, save_path="outputs/plots/pixel_frequency_per_class.png")
    print("Pixel frequency per class plot saved as outputs/plots/pixel_frequency_per_class.png")

    # Plot class appearance per image
    plot_class_appearance_per_image(class_appears_in, class_names, save_path="outputs/plots/class_appearance_per_image.png")
    print("Class appearance per image plot saved as outputs/plots/class_appearance_per_image.png")

    # Print detailed class analysis (textual)
    print_class_frequencies(
        pixel_counts, total_pixels, class_appears_in, labelids_list,
        id2label, MASK_SUFFIX, IMAGE_SUFFIX, MASK_DIR, IMG_DIR
    )

    # Export statistics to CSV and JSON
    export_statistics_csv_json(pixel_counts, total_pixels, class_appears_in, resolutions)


if __name__ == "__main__":
    buffer = TerminalToPDF("terminal_output.pdf")
    buffer.start()
    try:
        images = list_images(IMG_DIR, IMAGE_SUFFIX)
        print(f"Total images found: {len(images)}")

        # Get the resolutions of all images
        resolutions = get_resolutions(images)
        if resolutions:
            width, height = resolutions[0]
            print(f"Typical resolution: {width}x{height} (first image)")

            # Check if all have the same resolution
            unique_resolutions = set(resolutions)  # set = get unique elements
            print(f"Unique resolutions found: {unique_resolutions}")
        else:
            print("No resolutions found.")

        # Plot and save histogram of resolutions
        plot_resolution_histogram(resolutions)

        # Analyze classes and pixel frequencies
        labelids = list_labelids(MASK_DIR, MASK_SUFFIX)
        print(f"\nTotal labelIds masks found: {len(labelids)}")

        # Generate overlays for all image/mask pairs directly in outputs/overlays
        print("\nGenerating overlays for all masks...")
        generate_all_overlays(labelids, id2label, MASK_SUFFIX, IMAGE_SUFFIX, MASK_DIR, IMG_DIR)

        # Analyze classes and pixel frequencies in labelIds masks
        analyze_classes_and_frequencies(labelids, resolutions)
    finally:
        buffer.stop()
        buffer.save_pdf()

        # Generate PDF report after all analysis and plots are created
        generate_cityscapes_report()

