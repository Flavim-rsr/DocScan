"""
image_mode.py - Processamento de imagem estática (modo imagem)
Técnicas clássicas de Visão Computacional
"""

import cv2
import os
import sys
from src.scanner import detect_and_scan
from src.utils import save_results, resize_for_display


def process_image(image_path, output_dir="assets/output", show_preview=True):
    """
    Carrega uma imagem do disco, detecta e escaneia o documento nela.
    Salva os resultados e opcionalmente mostra uma comparação antes/depois.

    Parâmetros:
        image_path  : caminho para a imagem de entrada
        output_dir  : pasta de saída para os arquivos salvos
        show_preview: se True, abre janelas de comparação
    """
    # Verifica se o arquivo existe
    if not os.path.isfile(image_path):
        print(f"[ERRO] Arquivo não encontrado: {image_path}")
        return False

    print(f"[INFO] Carregando imagem: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERRO] Não foi possível ler a imagem: {image_path}")
        return False

    print("[INFO] Detectando documento...")
    warped_color, warped_gray, doc_pts = detect_and_scan(image)

    if warped_color is None:
        print("[AVISO] Nenhum documento retangular encontrado na imagem.")
        print("        Dica: certifique-se de que o documento ocupa boa parte da imagem.")
        return False

    # Salva os dois resultados
    path_color, path_gray = save_results(warped_color, warped_gray, output_dir)
    print(f"[OK] Documento colorido salvo em : {path_color}")
    print(f"[OK] Documento em cinza salvo em : {path_gray}")

    if show_preview:
        _show_comparison(image, doc_pts, warped_color, warped_gray)

    return True


def _show_comparison(original, doc_pts, warped_color, warped_gray):
    """
    Exibe apenas os dois resultados finais: colorido e preto-e-branco/cinza.
    Aguarda tecla para fechar.
    """
    # Redimensiona para exibição
    color_disp       = resize_for_display(warped_color)
    gray_disp        = resize_for_display(warped_gray)

    cv2.imshow("Resultado: Colorido Melhorado",  color_disp)
    cv2.imshow("Resultado: Preto e Branco",      gray_disp)

    print("\n[INFO] Pressione qualquer tecla nas janelas para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interactive_image_menu():
    """
    Menu interativo no terminal para escolher a imagem a processar.
    Lista arquivos disponíveis em assets/input/ e permite digitar o caminho.
    """
    print("\n" + "=" * 55)
    print("   MODO IMAGEM - Scanner de Documentos")
    print("=" * 55)

    input_dir = "assets/input"
    if os.path.isdir(input_dir):
        files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
        if files:
            print(f"\nImagens disponíveis em '{input_dir}':")
            for i, f in enumerate(files, 1):
                print(f"  [{i}] {f}")
        else:
            print(f"\n(Nenhuma imagem encontrada em '{input_dir}')")

    print("\nDigite o número da imagem acima ou o caminho completo:")
    path_input = input(">>> ").strip()

    # Permite digitar número para selecionar da lista
    if path_input.isdigit() and os.path.isdir(input_dir):
        idx = int(path_input) - 1
        if 0 <= idx < len(files):
            path_input = os.path.join(input_dir, files[idx])
        else:
            print("[ERRO] Número inválido.")
            return

    process_image(path_input)
