"""
utils.py - Funções utilitárias para o Scanner de Documentos
Técnicas clássicas de Visão Computacional
"""

import cv2
import numpy as np
import os
from datetime import datetime


def order_points(pts):
    """
    Ordena 4 pontos na ordem: [topo-esquerda, topo-direita, baixo-direita, baixo-esquerda].
    Necessário para aplicar a transformação de perspectiva corretamente.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # A soma dos pontos: topo-esquerda tem a menor soma, baixo-direita tem a maior
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # topo-esquerda
    rect[2] = pts[np.argmax(s)]  # baixo-direita

    # A diferença dos pontos: topo-direita tem a menor diferença, baixo-esquerda a maior
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # topo-direita
    rect[3] = pts[np.argmax(diff)]  # baixo-esquerda

    return rect


def four_point_transform(image, pts, doc_width_cm=None, doc_height_cm=None):
    """
    Aplica a transformação de perspectiva (warp) usando os 4 pontos do documento.
    Se doc_width_cm e doc_height_cm forem fornecidos, a saída é forçada para essa
    proporção exata (garante que o documento não saia deformado).
    Resultado: imagem com o documento "achatado" e retificado.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calcula largura e altura medidas na imagem
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top    = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    meas_width   = max(int(width_bottom), int(width_top))

    height_left  = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    meas_height  = max(int(height_left), int(height_right))

    # Se a proporção real do documento for conhecida, corrige as dimensões de saída
    if doc_width_cm and doc_height_cm:
        known_ratio = doc_width_cm / doc_height_cm
        meas_ratio  = meas_width / max(meas_height, 1)
        # Decide orientação e ajusta para a proporção correta
        if meas_ratio >= 1.0:  # paisagem
            out_width  = meas_width
            out_height = int(round(meas_width / known_ratio))
        else:                   # retrato
            out_height = meas_height
            out_width  = int(round(meas_height * known_ratio))
        max_width, max_height = out_width, out_height
    else:
        max_width, max_height = meas_width, meas_height

    # Define os 4 pontos de destino
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Calcula a matriz de transformação e aplica o warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def enhance_grayscale(image):
    """
    Gera uma versão em tons de cinza com contraste e nitidez melhorados.
    Mantém detalhes importantes de RG/CNH que seriam perdidos em uma
    binarização pura.
    """
    # Converte para escala de cinza se necessário
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Reduz ruído da webcam sem apagar completamente as bordas
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE melhora contraste local sem transformar tudo em preto/branco
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Nitidez leve para destacar linhas e texto do documento
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def enhance_color(image):
    """
    Melhora a versão colorida do documento:
    aumenta contraste e nitidez sem binarizar.
    """
    # Converte para LAB para melhorar luminosidade sem afetar cores
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE: equalização adaptativa de histograma no canal L (luminosidade)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Reconstrói a imagem com luminosidade melhorada
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Sharpening leve com kernel de nitidez
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def get_timestamp_name(prefix="scan", ext="jpg"):
    """Gera nome de arquivo com data e hora atual."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def ensure_output_dir(path="assets/output"):
    """Cria o diretório de saída se ele não existir."""
    os.makedirs(path, exist_ok=True)


def save_results(warped_color, warped_gray, output_dir="assets/output"):
    """
    Salva as duas versões do documento (colorida e cinza) em assets/output.
    Retorna os caminhos dos arquivos salvos.
    """
    ensure_output_dir(output_dir)

    name_color = get_timestamp_name("scan_color", "jpg")
    name_gray  = get_timestamp_name("scan_gray",  "jpg")

    path_color = os.path.join(output_dir, name_color)
    path_gray  = os.path.join(output_dir, name_gray)

    cv2.imwrite(path_color, warped_color)
    cv2.imwrite(path_gray,  warped_gray)

    return path_color, path_gray


def draw_overlay_text(frame, lines, start_y=30, color=(0, 255, 0)):
    """
    Desenha múltiplas linhas de texto na imagem (frame da webcam).
    Útil para mostrar instruções ao usuário.
    """
    for i, line in enumerate(lines):
        y = start_y + i * 28
        # Sombra preta para legibilidade
        cv2.putText(frame, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # Texto principal
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def resize_for_display(image, max_width=900, max_height=700):
    """
    Redimensiona imagem para exibição sem distorcer proporções.
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image
