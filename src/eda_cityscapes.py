import os
import sys
import numpy as np
from PIL import Image

# Adiciona o path do cityscapesScripts para importar labels
sys.path.append('/home/senai/Documentos/cityscapesScripts')
from cityscapesscripts.helpers.labels import id2label, labels


# path imagens cityscapes (RGB)
CITYSCAPES_IMG_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit"
# path máscaras labelIds
CITYSCAPES_LABELIDS_DIR = "/home/senai/Documentos/edge_segmentation/datasets/cityscapes/gtFine_trainvaltest/gtFine"

# Função para listar todas as imagens nos splits train, val, test
def listar_imagens(caminho_base):
    imagens = [] # Lista para armazenar dos paths das imagens
    for split in ["train", "val", "test"]: 
        split_dir = os.path.join(caminho_base, split) # path para o split atual (train, val ou test)
        
        # Verificar se o diretório existe
        if not os.path.isdir(split_dir):
            continue

        # Iterar sobre as cidades dentro do split
        for cidade in os.listdir(split_dir): 
            cidade_dir = os.path.join(split_dir, cidade) 

            # Verificar se o diretório da cidade existe
            if not os.path.isdir(cidade_dir):
                continue

            # Iterar sobre os arquivos dentro do diretório da cidade
            for arquivo in os.listdir(cidade_dir):
                if arquivo.endswith("_leftImg8bit.png"):
                    imagens.append(os.path.join(cidade_dir, arquivo))
    
    # Retorna lista de paths das imagens encontradas
    return imagens 

# Função para obter resoluções das imagens
def obter_resolucoes(lista_imagens):
    resolucoes = [] # Lista para armazenar as resoluções
    
    for img_path in lista_imagens:
        try:
            with Image.open(img_path) as img: # Abrir a imagem ->> PIL
                resolucoes.append(img.size)  # (largura, altura)
        except Exception as e:
            print(f"Erro ao abrir {img_path}: {e}")
    
    # Retorna lista de resoluçoes
    return resolucoes

# Função para listar todas as máscaras labelIds nos splits train, val, test
def listar_labelids(caminho_base):
    mascaras = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(caminho_base, split)
        if not os.path.isdir(split_dir):
            continue
        for cidade in os.listdir(split_dir):
            cidade_dir = os.path.join(split_dir, cidade) # path para o diretório da cidade dentro do split
            if not os.path.isdir(cidade_dir):
                continue
            for arquivo in os.listdir(cidade_dir):
                if arquivo.endswith("_gtFine_labelIds.png"): 
                    # arquivo termina em _gtFine_labelIds.png é uma máscara labelIds
                    mascaras.append(os.path.join(cidade_dir, arquivo))
    return mascaras

# Função para analisar classes presentes e frequências de pixels
def analisar_classes_e_frequencias(lista_labelids):
    pixel_counts = {} # Dicionário para contar pixels por classe (cls_id: count)
    total_pixels = 0
    for mask_path in lista_labelids:
        try:
            mask = np.array(Image.open(mask_path))
            # np.unique retorna os valores únicos 
            unique, counts = np.unique(mask, return_counts=True) 

            for cls_id, count in zip(unique, counts):
                # Atualiza a contagem de pixels para a classe cls_id
                pixel_counts[cls_id] = pixel_counts.get(cls_id, 0) + count
            total_pixels += mask.size
        except Exception as e:
            print(f"Erro ao abrir máscara {mask_path}: {e}")
    
    # Frequência absoluta e relativa
    print("\nFrequência absoluta e relativa de pixels por classe:")

    print(f"{'Classe':<20} {'ID':<5} {'Pixels':<12} {'Relativa (%)':<12}")
    print("-"*55)
    
    # Ordena por frequência absoluta (maior para menor)
    for cls_id, count in sorted(pixel_counts.items(), key=lambda x: -x[1]):
        label = id2label.get(cls_id)
        nome = label.name if label else str(cls_id)
        rel = 100 * count / total_pixels if total_pixels > 0 else 0
        print(f"{nome:<20} {cls_id:<5} {count:<12} {rel:<12.4f}")

    print(f"\nTotal de pixels analisados: {total_pixels}")
    print("\nLista de classes presentes:")
    presentes = [id2label[cls_id].name for cls_id in pixel_counts if cls_id in id2label]
    print(presentes)


if __name__ == "__main__":
    imagens = listar_imagens(CITYSCAPES_IMG_DIR)
    print(f"Total de imagens encontradas: {len(imagens)}")

    # passa a lista de imagens para obter as resoluçoes 
    resolucoes = obter_resolucoes(imagens)
    if resolucoes:
        largura, altura = resolucoes[0]
        print(f"Resolução típica: {largura}x{altura} (primeira imagem)")
        
        # Checar se todas têm a mesma resolução
        resolucoes_unicas = set(resolucoes) # set = pega os elementos únicos
        print(f"Resoluções únicas encontradas: {resolucoes_unicas}")
    else:
        print("Nenhuma resolução encontrada.")

    # Analis classes e frequências de pixels
    labelids = listar_labelids(CITYSCAPES_LABELIDS_DIR) # Lista de paths das máscaras labelIds
    print(f"\nTotal de máscaras labelIds encontradas: {len(labelids)}" )
    
    # Analisa classes e frequências de pixels nas máscaras labelIds
    analisar_classes_e_frequencias(labelids)
