import os
import json
import shutil
import csv
import numpy as np
from PIL import Image

from dataset_config import (
    CITYSCAPES_ROOT,
    IMG_DIR,
    MASK_DIR,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    PREPARED_ROOT,
    PREPARED_SPLITS,
    PREPARED_IMAGES,
    PREPARED_MASKS,
    PREPARED_METADATA,
)

import importlib
labels_mod = importlib.import_module('helpers.cityscapes_labels')
name2label = labels_mod.name2label


def load_class_mapping(config_path="configs/class_mapping_cityscapes.json"):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    mapping = {item['original']: item['reduced'] for item in cfg.get('mapping', [])}
    ignore_index = cfg.get('ignore_index', 255)
    background_label = cfg.get('background_label', 'Background/Ignore')
    return mapping, ignore_index, background_label


def build_reduced_palette():
    """Define numeric ids for the reduced classes.
    Background/Ignore is assigned to 255 by convention.
    """
    reduced_to_id = {
        'Pista trafegável': 0,
        'Faixas de trânsito': 1,
        'Veículos': 2,
        'Pedestres': 3,
        'Obstáculos': 4,
        'Background/Ignore': 255,
    }
    return reduced_to_id


def ensure_dirs():
    os.makedirs(PREPARED_ROOT, exist_ok=True)
    os.makedirs(os.path.join(PREPARED_ROOT, PREPARED_METADATA), exist_ok=True)
    for split in PREPARED_SPLITS:
        os.makedirs(os.path.join(PREPARED_ROOT, split, PREPARED_IMAGES), exist_ok=True)
        os.makedirs(os.path.join(PREPARED_ROOT, split, PREPARED_MASKS), exist_ok=True)


def prepare_dataset(config_path="configs/class_mapping_cityscapes.json", perform_stratified_split=False, split_seed=42, ratios=(0.7, 0.15, 0.15)):
    """Prepara o dataset remapeando máscaras para as classes reduzidas e copiando imagens.

    - Lê `config_path` com o mapeamento nome->nome reduzido
    - Converte IDs originais para IDs reduzidos (usa `helpers.cityscapes_labels`)
    - Salva máscaras remapeadas em `dataset_prepared/cityscapes/<split>/masks`
    - Copia imagens correspondentes para `dataset_prepared/cityscapes/<split>/images`
    - Escreve `class_mapping.json` em metadata com mapeamento numérico
    - Opcionalmente executa um split estratificado quando `perform_stratified_split=True`
    """
    mapping_name_to_name, ignore_index, background_label = load_class_mapping(config_path)
    reduced_to_id = build_reduced_palette()

    # Build original name -> reduced id mapping (use label names from helpers)
    origid_to_reducedid = {}
    for orig_name, reduced_name in mapping_name_to_name.items():
        label = name2label.get(orig_name)
        if label is None:
            print(f"Warning: original label name '{orig_name}' not found in helpers.cityscapes_labels")
            continue
        origid = label.id
        red_id = reduced_to_id.get(reduced_name, reduced_to_id[background_label])
        origid_to_reducedid[origid] = red_id

    ensure_dirs()

    # Walk through masks and remap
    for split in PREPARED_SPLITS:
        src_split_dir = os.path.join(MASK_DIR, split)
        if not os.path.isdir(src_split_dir):
            continue
        for city in os.listdir(src_split_dir):
            city_mask_dir = os.path.join(src_split_dir, city)
            if not os.path.isdir(city_mask_dir):
                continue
            for fname in os.listdir(city_mask_dir):
                if not fname.endswith(MASK_SUFFIX):
                    continue
                src_mask_path = os.path.join(city_mask_dir, fname)
                # corresponding image path
                img_fname = fname.replace(MASK_SUFFIX, IMAGE_SUFFIX)
                src_img_path = os.path.join(os.path.join(IMG_DIR, split, city), img_fname)

                # Destination dirs keeping city structure
                dst_mask_dir = os.path.join(PREPARED_ROOT, split, PREPARED_MASKS, city)
                dst_img_dir = os.path.join(PREPARED_ROOT, split, PREPARED_IMAGES, city)
                os.makedirs(dst_mask_dir, exist_ok=True)
                os.makedirs(dst_img_dir, exist_ok=True)

                dst_mask_path = os.path.join(dst_mask_dir, fname)
                dst_img_path = os.path.join(dst_img_dir, img_fname)

                # Copy image if exists
                if os.path.exists(src_img_path):
                    try:
                        if not os.path.exists(dst_img_path):
                            shutil.copy2(src_img_path, dst_img_path)
                    except Exception as e:
                        print(f"Warning copying image {src_img_path}: {e}")
                else:
                    print(f"Warning: image not found for mask {src_mask_path}")

                # Remap mask
                try:
                    mask = np.array(Image.open(src_mask_path))
                    new_mask = np.full_like(mask, fill_value=ignore_index, dtype=np.uint8)
                    for orig_id, red_id in origid_to_reducedid.items():
                        new_mask[mask == orig_id] = red_id
                    # Save
                    Image.fromarray(new_mask).save(dst_mask_path)
                except Exception as e:
                    print(f"Error remapping mask {src_mask_path}: {e}")

    # Write numeric mapping to metadata
    mapping_meta = {
        'reduced_to_id': reduced_to_id,
        'original_id_to_reduced_id': {str(k): int(v) for k, v in origid_to_reducedid.items()},
        'ignore_index': int(ignore_index),
        'description': 'Mapeamento numérico entre ids originais e ids reduzidos (Background/Ignore=255).'
    }
    meta_path = os.path.join(PREPARED_ROOT, PREPARED_METADATA, 'class_mapping.json')
    with open(meta_path, 'w') as mf:
        json.dump(mapping_meta, mf, indent=2)
    print(f"Prepared dataset saved under {PREPARED_ROOT}. Mapping metadata: {meta_path}")

    # After preparing, generate dataset-level statistics and split report
    try:
        generate_dataset_statistics_and_split_report()
    except Exception as e:
        print(f"Warning: failed to generate dataset statistics/split report: {e}")


def generate_dataset_statistics_and_split_report():
    """Compute statistics over the prepared dataset and write metadata and split_report.csv"""
    reduced_to_id = build_reduced_palette()
    id_to_reduced_name = {v: k for k, v in reduced_to_id.items()}

    stats = {}
    stats['classes'] = {}
    stats['total_pixels'] = 0
    split_report_rows = []

    for split in PREPARED_SPLITS:
        masks_base = os.path.join(PREPARED_ROOT, split, PREPARED_MASKS)
        images_base = os.path.join(PREPARED_ROOT, split, PREPARED_IMAGES)
        if not os.path.isdir(masks_base):
            continue
        for city in os.listdir(masks_base):
            city_mask_dir = os.path.join(masks_base, city)
            city_img_dir = os.path.join(images_base, city)
            if not os.path.isdir(city_mask_dir):
                continue
            for fname in os.listdir(city_mask_dir):
                mask_path = os.path.join(city_mask_dir, fname)
                # corresponding image
                img_fname = fname.replace(MASK_SUFFIX, IMAGE_SUFFIX)
                img_path = os.path.join(city_img_dir, img_fname)

                try:
                    mask = np.array(Image.open(mask_path))
                except Exception as e:
                    print(f"Error opening prepared mask {mask_path}: {e}")
                    continue

                unique, counts = np.unique(mask, return_counts=True)
                stats['total_pixels'] += mask.size

                classes_present = []
                for cls_id, cnt in zip(unique, counts):
                    cls_id_int = int(cls_id)
                    stats['classes'][cls_id_int] = stats['classes'].get(cls_id_int, 0) + int(cnt)
                    if cnt > 0:
                        classes_present.append(str(cls_id_int))

                # image resolution
                width = height = 0
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as im:
                            width, height = im.size
                    except Exception:
                        pass

                rel_path = os.path.relpath(mask_path, PREPARED_ROOT)
                split_report_rows.append({
                    'file': rel_path.replace(os.sep, '/'),
                    'split': split,
                    'width': width,
                    'height': height,
                    'classes_present': ','.join(classes_present)
                })

    # Build classes list for JSON
    classes_list = []
    for cls_id, pixel_count in stats['classes'].items():
        name = id_to_reduced_name.get(cls_id, str(cls_id))
        classes_list.append({
            'class_id': int(cls_id),
            'class_name': name,
            'pixel_count': int(pixel_count),
            'pixel_percent': 100 * pixel_count / stats['total_pixels'] if stats['total_pixels'] else 0,
            'appears_in_images': None
        })

    dataset_stats = {
        'classes': classes_list,
        'total_pixels': int(stats['total_pixels']),
    }

    # Write dataset_statistics.json and split_report.csv into metadata
    meta_stats_path = os.path.join(PREPARED_ROOT, PREPARED_METADATA, 'dataset_statistics.json')
    with open(meta_stats_path, 'w') as jf:
        json.dump(dataset_stats, jf, indent=2)

    csv_path = os.path.join(PREPARED_ROOT, PREPARED_METADATA, 'split_report.csv')
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['file', 'split', 'width', 'height', 'classes_present'])
        for row in split_report_rows:
            writer.writerow([row['file'], row['split'], row['width'], row['height'], row['classes_present']])

    print(f"Wrote dataset statistics to {meta_stats_path} and split report to {csv_path}")


def stratified_split_dataset(seed=42, ratios=(0.7, 0.15, 0.15)):
    """Perform a simple stratified split over prepared images using reduced class presence.

    Grouping strategy: for each image, compute the set of reduced class ids present in its prepared mask.
    Use the smallest reduced id as a stratification key (background-only images get key 'bg').
    Then split each group proportionally according to `ratios` and move files to new split folders.
    """
    import random

    random.seed(seed)
    train_ratio, val_ratio, test_ratio = ratios

    # gather all prepared image/mask pairs
    items = []  # list of dicts: {'img': path, 'mask': path, 'city': city, 'fname': fname}
    for split in PREPARED_SPLITS:
        img_base = os.path.join(PREPARED_ROOT, split, PREPARED_IMAGES)
        mask_base = os.path.join(PREPARED_ROOT, split, PREPARED_MASKS)
        if not os.path.isdir(img_base):
            continue
        for city in os.listdir(img_base):
            city_img_dir = os.path.join(img_base, city)
            city_mask_dir = os.path.join(mask_base, city)
            if not os.path.isdir(city_img_dir):
                continue
            for fname in os.listdir(city_img_dir):
                if not fname.endswith(IMAGE_SUFFIX):
                    continue
                img_path = os.path.join(city_img_dir, fname)
                mask_fname = fname.replace(IMAGE_SUFFIX, MASK_SUFFIX)
                mask_path = os.path.join(city_mask_dir, mask_fname)
                if not os.path.exists(mask_path):
                    # try to find mask anywhere under PREPARED_ROOT
                    for root, _, files in os.walk(os.path.join(PREPARED_ROOT)):
                        if mask_fname in files:
                            mask_path = os.path.join(root, mask_fname)
                            break
                items.append({'img': img_path, 'mask': mask_path, 'city': city, 'fname': fname})

    # load reduced id set for detection
    mapping_path = os.path.join(PREPARED_ROOT, PREPARED_METADATA, 'class_mapping.json')
    ignore_idx = 255
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as mf:
            mm = json.load(mf)
            ignore_idx = int(mm.get('ignore_index', 255))

    # compute per-image reduced-classes present
    img_classes = []  # parallel list to items, each entry is set of reduced class ids present
    class_image_count = {}  # reduced_id -> number of images containing it
    for it in items:
        mask_path = it['mask']
        present = set()
        if os.path.exists(mask_path):
            try:
                mask = np.array(Image.open(mask_path))
                unique = np.unique(mask)
                for u in unique:
                    u_int = int(u)
                    if u_int == ignore_idx:
                        continue
                    present.add(u_int)
            except Exception:
                pass
        img_classes.append(present)
        for c in present:
            class_image_count[c] = class_image_count.get(c, 0) + 1

    total_images = len(items)

    # desired number of images per class per split
    desired = {}
    for cls_id, total_img_count in class_image_count.items():
        desired[cls_id] = {
            'train': int(round(total_img_count * train_ratio)),
            'val': int(round(total_img_count * val_ratio)),
            'test': int(round(total_img_count * test_ratio)),
        }

    # prepare destination dirs (clear existing train/val/test under PREPARED_ROOT)
    for split in PREPARED_SPLITS:
        for sub in [PREPARED_IMAGES, PREPARED_MASKS]:
            dst = os.path.join(PREPARED_ROOT, split, sub)
            if os.path.exists(dst):
                # remove existing to avoid mixing
                shutil.rmtree(dst)
            os.makedirs(dst, exist_ok=True)

    # greedy assignment: iterate images in random order, assign to split that best improves balance
    indices = list(range(total_images))
    random.shuffle(indices)

    # track current counts per class per split
    current = {cls_id: {'train': 0, 'val': 0, 'test': 0} for cls_id in class_image_count}
    assignments = {'train': [], 'val': [], 'test': []}

    for idx in indices:
        present = img_classes[idx]
        # if no present classes, assign to train by default
        if not present:
            assignments['train'].append(items[idx])
            continue

        best_split = None
        best_score = None
        for split in ['train', 'val', 'test']:
            score = 0.0
            for c in present:
                need = desired.get(c, {}).get(split, 0)
                cur = current.get(c, {}).get(split, 0)
                # ratio of fulfillment (higher means closer to desired)
                if need > 0:
                    score += (cur + 1) / float(need)  # +1 as if we assign here
                else:
                    # if no desired (rare), prefer not to overload
                    score += (cur + 1) / (1.0)
            # normalize by number of classes to avoid bias
            score = score / max(1, len(present))
            if best_score is None or score < best_score:
                best_score = score
                best_split = split

        assignments[best_split].append(items[idx])
        # update current counts
        for c in present:
            current[c][best_split] += 1

    # move files into new split structure
    for split, itm_list in assignments.items():
        for it in itm_list:
            city = it['city']
            fname = it['fname']
            img_src = it['img']
            mask_src = it['mask']
            dst_img_dir = os.path.join(PREPARED_ROOT, split, PREPARED_IMAGES, city)
            dst_mask_dir = os.path.join(PREPARED_ROOT, split, PREPARED_MASKS, city)
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_mask_dir, exist_ok=True)
            dst_img = os.path.join(dst_img_dir, fname)
            dst_mask = os.path.join(dst_mask_dir, fname.replace(IMAGE_SUFFIX, MASK_SUFFIX))
            try:
                if os.path.exists(img_src):
                    shutil.move(img_src, dst_img)
                if os.path.exists(mask_src):
                    shutil.move(mask_src, dst_mask)
            except Exception as e:
                print(f"Warning moving files for {fname}: {e}")

    # write split metadata
    split_meta = {'seed': int(seed), 'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}}
    with open(os.path.join(PREPARED_ROOT, PREPARED_METADATA, 'split_metadata.json'), 'w') as sf:
        json.dump(split_meta, sf, indent=2)

    print(f"Stratified split completed (seed={seed}). New splits under {PREPARED_ROOT}")


if __name__ == '__main__':
    prepare_dataset()
