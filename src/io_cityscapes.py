import os
from PIL import Image

def list_images(base_path, image_suffix):
    '''Lista os caminhos completos das imagens RGB no formato cityscapes'''
    images = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_path, split)
        if not os.path.isdir(split_dir):
            continue
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for filename in os.listdir(city_dir):
                if filename.endswith(image_suffix):
                    images.append(os.path.join(city_dir, filename))
    return images


def list_prepared_images(prepared_root, image_suffix):
    """Lista imagens no layout dataset_prepared/<split>/images/<city>/*.png"""
    images = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(prepared_root, split, "images")
        if not os.path.isdir(split_dir):
            continue
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for filename in os.listdir(city_dir):
                if filename.endswith(image_suffix):
                    images.append(os.path.join(city_dir, filename))
    return images


def list_prepared_labelids(prepared_root, mask_suffix):
    """Lista máscaras no layout dataset_prepared/<split>/masks/<city>/*_gtFine_labelIds.png"""
    masks = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(prepared_root, split, "masks")
        if not os.path.isdir(split_dir):
            continue
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for filename in os.listdir(city_dir):
                if filename.endswith(mask_suffix):
                    masks.append(os.path.join(city_dir, filename))
    return masks

def list_labelids(base_path, mask_suffix):
    '''Lista os caminhos completos dos arquivos de máscara de rótulos (labelIds) no formato cityscapes'''
    masks = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_path, split)
        if not os.path.isdir(split_dir):
            continue
        for city in os.listdir(split_dir):
            city_dir = os.path.join(split_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for filename in os.listdir(city_dir):
                if filename.endswith(mask_suffix):
                    masks.append(os.path.join(city_dir, filename))
    return masks

def get_resolutions(image_list):
    '''Obtém as resoluções (largura, altura) de uma lista de imagens.'''
    resolutions = []
    for img_path in image_list:
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.size)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
    return resolutions
