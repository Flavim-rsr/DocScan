"""
main.py - Ponto de entrada do Scanner de Documentos
Técnicas clássicas de Visão Computacional

Uso:
    python src/main.py                  -> menu interativo
    python src/main.py --webcam         -> modo webcam direto
    python src/main.py --image foto.jpg -> modo imagem direto
"""

import sys
import os
import cv2
import numpy as np

# Adiciona o diretório raiz ao path para imports funcionarem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scanner import detect_and_scan, draw_document_outline
from src.utils import (save_results, draw_overlay_text, resize_for_display,
                       ensure_output_dir)
from src.image_mode import process_image, interactive_image_menu


# ─────────────────────────────────────────────
#  MODO WEBCAM
# ─────────────────────────────────────────────

def run_webcam_mode(output_dir="assets/output"):
    """
    Abre a webcam, detecta documentos em tempo real e permite salvar com 'S'.
    Teclas:
      S - salva documento detectado (colorido + cinza)
      Q - sai da aplicação
    """
    ensure_output_dir(output_dir)

    print("\n[INFO] Abrindo webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a webcam.")
        print("       Verifique se ela está conectada e não está sendo usada por outro programa.")
        return

    print("[INFO] Webcam aberta com sucesso.")
    print("[INFO] Aponte a câmera para um documento sobre fundo contrastante.")
    print("[INFO] Teclas: [S] salvar  |  [Q] sair\n")

    # Estado da última detecção bem-sucedida (para salvar ao pressionar S)
    last_warped_color = None
    last_warped_gray  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao capturar frame da webcam.")
            break

        # Copia para exibição (não queremos modificar o frame original)
        display_frame = frame.copy()

        # ── Detecção ──────────────────────────────────────────
        warped_color, warped_gray, doc_pts = detect_and_scan(frame)

        if warped_color is not None:
            last_warped_color = warped_color
            last_warped_gray  = warped_gray

            # Desenha contorno verde sobre o documento
            draw_document_outline(display_frame, doc_pts, color=(0, 255, 0))

            status_text  = "Documento detectado!"
            status_color = (0, 220, 0)
        else:
            status_text   = "Buscando documento..."
            status_color  = (0, 165, 255)

        # ── Instruções na tela ────────────────────────────────
        instructions = [
            f"Status: {status_text}",
            "S = Salvar documento",
            "Q = Sair",
        ]
        draw_overlay_text(display_frame, instructions, start_y=30, color=status_color)

        # Exibe frame principal
        main_display = resize_for_display(display_frame, max_width=900, max_height=650)
        cv2.imshow("Scanner de Documentos - Webcam", main_display)

        # ── Teclas ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("[INFO] Encerrando...")
            break

        elif key == ord('s') or key == ord('S'):
            if last_warped_color is not None and last_warped_gray is not None:
                path_color, path_gray = save_results(
                    last_warped_color, last_warped_gray, output_dir
                )
                print(f"[OK] Salvo: {path_color}")
                print(f"[OK] Salvo: {path_gray}")

                # Feedback visual: flash verde no frame
                flash = np.zeros_like(display_frame)
                flash[:] = (0, 200, 0)
                cv2.addWeighted(display_frame, 0.7, flash, 0.3, 0, display_frame)
                cv2.imshow("Scanner de Documentos - Webcam",
                           resize_for_display(display_frame))
                cv2.waitKey(300)
            else:
                print("[AVISO] Nenhum documento detectado para salvar.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam encerrada.")


# ─────────────────────────────────────────────
#  MENU PRINCIPAL
# ─────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 55)
    print("   SCANNER DE DOCUMENTOS - Visão Computacional")
    print("   Projeto de Computação Gráfica  |  OpenCV")
    print("=" * 55)


def main_menu():
    """Menu interativo para escolher o modo de operação."""
    print_banner()
    print("\nEscolha o modo:")
    print("  [1] Modo Webcam  - escanear em tempo real")
    print("  [2] Modo Imagem  - processar imagem do disco")
    print("  [0] Sair")
    print()

    choice = input(">>> ").strip()

    if choice == "1":
        run_webcam_mode()
    elif choice == "2":
        interactive_image_menu()
    elif choice == "0":
        print("[INFO] Encerrando.")
        sys.exit(0)
    else:
        print("[AVISO] Opção inválida. Tente novamente.")
        main_menu()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--webcam" in args:
        run_webcam_mode()

    elif "--image" in args:
        idx = args.index("--image")
        if idx + 1 < len(args):
            image_path = args[idx + 1]
            process_image(image_path)
        else:
            print("[ERRO] Informe o caminho da imagem após --image")
            print("       Exemplo: python src/main.py --image assets/input/foto.jpg")
            sys.exit(1)

    else:
        main_menu()
