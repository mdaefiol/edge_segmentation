import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
 
def print_class_frequencies(pixel_counts, total_pixels, class_appears_in, labelids_list, id2label, MASK_SUFFIX, IMAGE_SUFFIX, MASK_DIR, IMG_DIR):
    '''Prints class frequencies and the overlay of classes present in the masks'''
    ignored_pixels = 0
    for cls_id in pixel_counts:
        if cls_id in id2label:
            label = id2label[cls_id]
            if hasattr(label, 'ignoreInEval') and label.ignoreInEval:
                ignored_pixels += pixel_counts[cls_id]
    if total_pixels > 0:
        perc_ignored = 100 * ignored_pixels / total_pixels
    else:
        perc_ignored = 0
    print(f"\nIgnored/unlabeled pixels: {ignored_pixels} ({perc_ignored:.2f}% of total)")

    # image and mask overlay
    overlay_created = False
    for example_mask_path in labelids_list:
        img_path = example_mask_path.replace(MASK_SUFFIX, IMAGE_SUFFIX)
        if MASK_DIR in example_mask_path and IMG_DIR not in example_mask_path:
            img_path = example_mask_path.replace(MASK_DIR, IMG_DIR)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(example_mask_path)
            img_np = np.array(img)
            mask_np = np.array(mask)
            color_mask = np.zeros_like(img_np)
            for cls_id in id2label:
                label = id2label[cls_id]
                color = label.color
                mask_area = (mask_np == cls_id)
                color_mask[mask_area] = color
            overlay = img_np.copy()
            alpha = 0.4
            overlay = (img_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_np)
            axs[0].set_title("Original Image")
            axs[0].axis("off")
            axs[1].imshow(mask_np, cmap="nipy_spectral")
            axs[1].set_title("Mask (labelIds)")
            axs[1].axis("off")
            axs[2].imshow(overlay)
            axs[2].set_title("Overlay")
            axs[2].axis("off")
            plt.tight_layout()
            plt.savefig("example_overlay.png")
            plt.close()
            print("Example image and mask overlay saved as example_overlay.png")
            overlay_created = True
            break
        except Exception as e:
            continue
    if not overlay_created:
        print("Could not create overlay example: No matching image/mask pair found.")

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
        percent = 100 * freq / num_imgs if num_imgs else 0
        print(f"{name:<20} {cls_id:<5} {freq:<8} ({percent:.2f}% of images)")

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

def plot_pixel_frequency_per_class(class_pixel_counts, class_names, save_path=None):
    '''Plots a bar chart of pixel frequency per class'''
    
    ids = list(class_pixel_counts.keys())
    counts = [class_pixel_counts[i] for i in ids]
    names = [class_names[i] for i in ids]

    plt.figure(figsize=(12, 6))
    plt.bar(names, counts, color='skyblue')
    plt.xlabel('Classe')
    plt.ylabel('Frequência de pixels')
    plt.title('Frequência de pixels por classe')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_class_appearance_per_image(class_appearance_counts, class_names, save_path=None):
    '''Plots a bar chart of the number of images in which each class appears'''
    ids = list(class_appearance_counts.keys())
    counts = [class_appearance_counts[i] for i in ids]
    names = [class_names[i] for i in ids]

    plt.figure(figsize=(12, 6))
    plt.bar(names, counts, color='orange')
    plt.xlabel('Classe')
    plt.ylabel('Número de imagens com a classe')
    plt.title('Aparição das classes por imagem')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
