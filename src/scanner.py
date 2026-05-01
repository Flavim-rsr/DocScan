"""
scanner.py - Pipeline principal de detecção e escaneamento de documentos
Técnicas clássicas de Visão Computacional

Estratégia de detecção em dois estágios:
  1. Método primário  : busca de contornos quadrilaterais (rápido)
  2. Método fallback  : Transformada de Hough (robusto para fundos claros
                        e documentos segurados na mão)
"""

import cv2
import numpy as np
from src.utils import order_points, four_point_transform, enhance_grayscale, enhance_color

# Proporção conhecida do documento: 8,5cm × 6cm
# O sistema aceita o documento em paisagem (8.5/6) ou retrato (6/8.5)
DOC_RATIO_LANDSCAPE = 8.5 / 6.0   # ≈ 1.417
DOC_RATIO_PORTRAIT  = 6.0 / 8.5   # ≈ 0.706
DOC_RATIO_TOLERANCE = 0.30         # ±30% de tolerância


# ─────────────────────────────────────────────────────────────
#  REDIMENSIONAMENTO
# ─────────────────────────────────────────────────────────────

def resize_with_ratio(image, height=800):
    h, w = image.shape[:2]
    ratio = height / h
    resized = cv2.resize(image, (int(w * ratio), height))
    return resized, ratio


# ─────────────────────────────────────────────────────────────
#  PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────

def preprocess_for_edges(image):
    """
    Pré-processamento robusto para ambientes com baixo contraste:
    1. CLAHE no canal L (melhora contraste local, ótimo para fundo claro)
    2. Escala de cinza
    3. Bilateral Filter (preserva bordas, remove ruído de câmera)
    4. Auto-Canny adaptativo (thresholds calculados pela mediana)
    5. Fechamento morfológico (fecha brechas no contorno)
    """
    # CLAHE no espaço LAB para melhorar contraste sem saturar cores
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image_eq = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(image_eq, cv2.COLOR_BGR2GRAY)

    # Bilateral: suaviza sem destruir bordas do documento
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Auto-Canny: thresholds baseados na mediana da imagem
    median = np.median(blurred)
    sigma  = 0.33
    lower  = int(max(0,   (1.0 - sigma) * median))
    upper  = int(min(255, (1.0 + sigma) * median))
    edges  = cv2.Canny(blurred, lower, upper)

    # Fechamento morfológico: dilata + erode para fechar lacunas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


# ─────────────────────────────────────────────────────────────
#  DETECÇÃO POR REGIÃO CLARA
# ─────────────────────────────────────────────────────────────

def preprocess_for_document_mask(image):
    """
    Cria uma máscara para documentos claros sobre fundo escuro.
    Ajuda quando a borda externa aparece fraca no Canny, mas o corpo do
    documento ainda contrasta bem com a mesa/fundo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _quad_fill_ratio(contour, quad):
    """Mede quanto do retângulo candidato é preenchido pelo contorno real."""
    quad_area = cv2.contourArea(quad.astype(np.float32))
    if quad_area < 1:
        return 0
    return cv2.contourArea(contour) / quad_area


def find_bright_document_quad(image, min_area_ratio=0.04):
    """
    Procura o documento como a maior região clara retangular.
    É útil para RG/CNH/cartões sobre fundo preto ou bem contrastante.
    """
    mask = preprocess_for_document_mask(image)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    h, w = mask.shape[:2]
    min_area = h * w * min_area_ratio
    candidates = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype("float32")

        if not _is_valid_quad(box.reshape(4, 1, 2), min_area):
            continue

        fill_ratio = _quad_fill_ratio(contour, box)
        if fill_ratio < 0.55:
            continue

        candidates.append((area * fill_ratio, box))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# ─────────────────────────────────────────────────────────────
#  MÉTODO 1 — CONTORNOS
# ─────────────────────────────────────────────────────────────

def _quad_aspect_ratio(pts):
    """
    Calcula a proporção largura/altura de um quadrilátero ordenado.
    Usa a média das larguras e alturas opostas.
    """
    ordered = order_points(pts.reshape(4, 2).astype("float32"))
    tl, tr, br, bl = ordered
    width  = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if height < 1:
        return 0
    return width / height


def _is_valid_quad(approx, min_area, check_ratio=True):
    if len(approx) != 4:
        return False
    if cv2.contourArea(approx) < min_area:
        return False
    if not cv2.isContourConvex(approx):
        return False
    if check_ratio:
        ratio = _quad_aspect_ratio(approx)
        lo_l = DOC_RATIO_LANDSCAPE * (1 - DOC_RATIO_TOLERANCE)
        hi_l = DOC_RATIO_LANDSCAPE * (1 + DOC_RATIO_TOLERANCE)
        lo_p = DOC_RATIO_PORTRAIT  * (1 - DOC_RATIO_TOLERANCE)
        hi_p = DOC_RATIO_PORTRAIT  * (1 + DOC_RATIO_TOLERANCE)
        if not (lo_l <= ratio <= hi_l or lo_p <= ratio <= hi_p):
            return False
    return True


def find_contour_quad(edges, min_area_ratio=0.03):
    """
    Método primário: busca o maior contorno quadrilateral.
    Testa múltiplos valores de epsilon para approxPolyDP.
    Fallback: convex hull do maior contorno.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = edges.shape[0] * edges.shape[1] * min_area_ratio
    epsilons = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]

    # Testa os 15 maiores contornos
    for contour in contours[:15]:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for eps in epsilons:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if _is_valid_quad(approx, min_area):
                return approx.reshape(4, 2).astype("float32")

    # Fallback: convex hull
    for contour in contours[:5]:
        if cv2.contourArea(contour) < min_area:
            break
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for eps in epsilons:
            approx = cv2.approxPolyDP(hull, eps * perimeter, True)
            if _is_valid_quad(approx, min_area):
                return approx.reshape(4, 2).astype("float32")

    return None


# ─────────────────────────────────────────────────────────────
#  MÉTODO 2 — HOUGH LINES (fallback robusto)
# ─────────────────────────────────────────────────────────────

def _line_intersection(line1, line2):
    """Calcula o ponto de interseção entre duas linhas no formato (rho, theta)."""
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])

    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None  # Linhas paralelas

    x, y = np.linalg.solve(A, b)
    return (float(x), float(y))


def find_hough_quad(edges, image_shape):
    """
    Método fallback: usa a Transformada de Hough Probabilística para
    encontrar as 4 bordas dominantes do documento e calcular seus cantos.

    Funciona bem em cenários onde o contorno do documento está incompleto
    (mão cobrindo cantos, reflexos, fundo de contraste similar).
    """
    h, w = image_shape[:2]

    # Hough Probabilístico: encontra segmentos de linha
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=80,
        maxLineGap=20
    )

    if lines is None or len(lines) < 4:
        return None

    # Converte segmentos para formato (rho, theta)
    rho_theta = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 50:
            continue
        # Ângulo da linha
        angle = np.arctan2(dy, dx)
        # Normaliza para [0, pi)
        if angle < 0:
            angle += np.pi
        # Rho: distância da origem
        rho = x1 * np.cos(angle - np.pi/2) + y1 * np.sin(angle - np.pi/2)
        rho_theta.append((rho, angle - np.pi/2, length))

    if len(rho_theta) < 4:
        return None

    # Separa linhas em horizontais e verticais
    horizontals = [(rho, theta, l) for rho, theta, l in rho_theta
                   if abs(np.sin(theta)) > 0.7]
    verticals   = [(rho, theta, l) for rho, theta, l in rho_theta
                   if abs(np.cos(theta)) > 0.7]

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    # Ordena por rho para pegar as bordas externas (top/bottom, left/right)
    horizontals.sort(key=lambda x: x[0])
    verticals.sort(key=lambda x: x[0])

    top    = horizontals[0][:2]
    bottom = horizontals[-1][:2]
    left   = verticals[0][:2]
    right  = verticals[-1][:2]

    # Calcula os 4 cantos do documento
    corners = []
    for h_line in [top, bottom]:
        for v_line in [left, right]:
            pt = _line_intersection(h_line, v_line)
            if pt is not None:
                corners.append(pt)

    if len(corners) != 4:
        return None

    pts = np.array(corners, dtype="float32")

    # Valida que os pontos estão dentro da imagem (com margem)
    margin = -50
    if not all(
        margin <= x <= w - margin and margin <= y <= h - margin
        for x, y in pts
    ):
        return None

    # Valida área mínima
    hull = cv2.convexHull(pts.astype(np.int32))
    area = cv2.contourArea(hull)
    if area < h * w * 0.03:
        return None

    return pts


# ─────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def detect_and_scan(image):
    """
    Pipeline completa de detecção e escaneamento.
    Tenta o método de contornos primeiro; se falhar, usa Hough Lines.

    Retorna:
      warped_color : documento colorido corrigido e melhorado
      warped_gray  : documento em tons de cinza melhorado
      doc_pts      : 4 pontos do contorno na escala original (ou None)
    """
    original = image.copy()

    # Redimensiona para processamento consistente
    resized, ratio = resize_with_ratio(original, height=800)

    # ── Método 0: região clara sobre fundo escuro ────────────
    doc_pts = find_bright_document_quad(resized)

    # ── Método 1: contornos ──────────────────────────────────
    if doc_pts is None:
        # Pré-processamento com CLAHE + Bilateral + Auto-Canny + Fechamento
        edges = preprocess_for_edges(resized)
        doc_pts = find_contour_quad(edges)

    # ── Método 2: Hough Lines (fallback) ─────────────────────
    if doc_pts is None:
        if "edges" not in locals():
            edges = preprocess_for_edges(resized)
        doc_pts = find_hough_quad(edges, resized.shape)

    if doc_pts is None:
        return None, None, None

    # Converte pontos para escala original
    doc_pts_original = doc_pts / ratio

    # Transformação de perspectiva com proporção real do documento (8,5 × 6 cm)
    warped = four_point_transform(original, doc_pts_original,
                                  doc_width_cm=8.5, doc_height_cm=6.0)

    warped_color = enhance_color(warped)
    warped_gray  = enhance_grayscale(warped)

    return warped_color, warped_gray, doc_pts_original


# ─────────────────────────────────────────────────────────────
#  VISUALIZAÇÃO
# ─────────────────────────────────────────────────────────────

def draw_document_outline(frame, doc_pts, ratio=1.0, color=(0, 255, 0), thickness=3):
    """Desenha o contorno do documento detectado no frame."""
    if doc_pts is None:
        return frame

    # IMPORTANTE: ordena os pontos antes de desenhar para evitar linhas cruzadas
    ordered = order_points(doc_pts)
    pts = (ordered * ratio).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    for pt in ordered:
        x, y = int(pt[0] * ratio), int(pt[1] * ratio)
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

    return frame
