"""
modules/image_processor.py
Pipeline de traitement d'image : redressement perspective + OCR Tesseract
pour extraire le Nom et le Numéro d'une carte Pokémon française.
"""

import cv2
import numpy as np
import pytesseract
import re
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OcrResult:
    """Résultat brut de l'extraction OCR."""
    raw_name: str = ""
    raw_number: str = ""
    full_text: str = ""
    parsed_name: str = ""
    parsed_number: str = ""
    hp_detected: Optional[int] = None
    confidence_score: float = 0.0


class ImageProcessor:
    """
    Prétraitement d'image et extraction OCR pour les cartes Pokémon.

    Pipeline :
        1. Conversion en tableau numpy
        2. Redressement de perspective (détection des 4 coins de la carte)
        3. Amélioration luminosité / contraste / netteté
        4. Découpe des zones clés (nom, numéro)
        5. OCR Tesseract avec config optimisée
        6. Post-traitement / nettoyage du texte
    """

    # Config Tesseract : OEM 3 (LSTM), PSM 6 (bloc de texte uniforme)
    _OCR_CONFIG_BLOCK = "--oem 3 --psm 6 -l fra+eng"
    # PSM 7 = ligne unique (idéal pour le nom ou le numéro)
    _OCR_CONFIG_LINE = "--oem 3 --psm 7 -l fra+eng"
    # Ratio standard carte Pokémon (largeur / hauteur)
    _CARD_RATIO = 2.5 / 3.5

    def process(self, img_array: np.ndarray) -> tuple[np.ndarray, OcrResult]:
        """
        Point d'entrée principal.

        Args:
            img_array: Image en RGB (numpy array HxWx3).

        Returns:
            (processed_img, OcrResult) — image traitée + données OCR.
        """
        # 1. Redresser la perspective si possible
        corrected = self._correct_perspective(img_array)

        # 2. Améliorer la qualité
        enhanced = self._enhance_image(corrected)

        # 3. Extraire les métadonnées par OCR
        ocr_result = self._extract_metadata(enhanced)

        return enhanced, ocr_result

    # ─────────────────────────────────────────────
    # Perspective
    # ─────────────────────────────────────────────

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """
        Détecte les bords de la carte et applique une transformation
        de perspective pour obtenir une vue orthogonale.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Dilatation légère pour fermer les contours ouverts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img

        # Garder le plus grand contour (la carte)
        largest = max(contours, key=cv2.contourArea)

        # Filtrer les contours trop petits (bruit)
        img_area = img.shape[0] * img.shape[1]
        if cv2.contourArea(largest) < img_area * 0.1:
            return img

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            warped = self._four_point_transform(img, pts)
            return warped

        return img

    def _four_point_transform(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Warp perspective vers une vue 2D propre de la carte."""
        # Ordonner : haut-gauche, haut-droit, bas-droit, bas-gauche
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # haut-gauche
        rect[2] = pts[np.argmax(s)]   # bas-droit
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # haut-droit
        rect[3] = pts[np.argmax(diff)]  # bas-gauche

        # Dimensions de sortie (ratio carte Pokémon standard)
        out_w = 500
        out_h = int(out_w / self._CARD_RATIO)

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img, M, (out_w, out_h))

    # ─────────────────────────────────────────────
    # Enhancement
    # ─────────────────────────────────────────────

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Améliore contraste, netteté et luminosité via PIL.
        Ajuste dynamiquement selon la luminosité moyenne détectée.
        """
        pil = Image.fromarray(img)

        # Luminosité moyenne
        gray_np = np.mean(np.array(pil.convert("L")))

        # Correction adaptative
        brightness_factor = 1.15 if gray_np < 128 else 1.0
        contrast_factor = 1.6 if gray_np < 100 else 1.3

        pil = ImageEnhance.Brightness(pil).enhance(brightness_factor)
        pil = ImageEnhance.Contrast(pil).enhance(contrast_factor)
        pil = ImageEnhance.Sharpness(pil).enhance(2.2)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        return np.array(pil)

    # ─────────────────────────────────────────────
    # OCR
    # ─────────────────────────────────────────────

    def _extract_metadata(self, img: np.ndarray) -> OcrResult:
        """
        Découpe les zones d'intérêt et lance l'OCR sur chacune.

        Zones (hauteur relative) :
            - Nom      : 0 – 14%   (bande en haut, hors HP)
            - Numéro   : 89 – 100% (bande en bas à gauche)
            - HP       : 0 – 14%   (partie droite de la bande du haut)
        """
        h, w = img.shape[:2]

        name_region   = img[0         : int(h * 0.14), 0         : int(w * 0.75)]
        number_region = img[int(h * 0.89) : h,         0         : int(w * 0.55)]
        hp_region     = img[0         : int(h * 0.14), int(w * 0.55) : w       ]

        name_raw   = self._ocr_region(name_region,   config=self._OCR_CONFIG_LINE)
        number_raw = self._ocr_region(number_region, config=self._OCR_CONFIG_LINE)
        hp_raw     = self._ocr_region(hp_region,     config=self._OCR_CONFIG_LINE)
        full_text  = self._ocr_full(img)

        parsed_name   = self._parse_name(name_raw, full_text)
        parsed_number = self._parse_number(number_raw, full_text)
        hp_val        = self._parse_hp(hp_raw)

        # Score de confiance basé sur ce qu'on a pu extraire
        confidence = 0.0
        if parsed_name:  confidence += 50.0
        if parsed_number: confidence += 40.0
        if hp_val:        confidence += 10.0

        return OcrResult(
            raw_name=name_raw,
            raw_number=number_raw,
            full_text=full_text,
            parsed_name=parsed_name,
            parsed_number=parsed_number,
            hp_detected=hp_val,
            confidence_score=confidence,
        )

    def _ocr_region(self, region: np.ndarray, config: str) -> str:
        """OCR sur une région avec upscaling pour améliorer la précision."""
        if region is None or region.size == 0:
            return ""
        pil = Image.fromarray(region)
        # Upscale ×3 pour de meilleures performances Tesseract
        pil = pil.resize(
            (max(pil.width * 3, 60), max(pil.height * 3, 20)),
            Image.LANCZOS,
        )
        # Améliorer le contraste local pour l'OCR
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        return pytesseract.image_to_string(pil, config=config).strip()

    def _ocr_full(self, img: np.ndarray) -> str:
        """OCR complet sur toute la carte (fallback)."""
        pil = Image.fromarray(img)
        return pytesseract.image_to_string(pil, config=self._OCR_CONFIG_BLOCK).strip()

    # ─────────────────────────────────────────────
    # Parsers
    # ─────────────────────────────────────────────

    def _parse_name(self, name_text: str, full_text: str) -> str:
        """
        Extrait le nom du Pokémon (première ligne significative, capitalisée).
        Retire les artefacts OCR courants (chiffres parasites, ponctuation).
        """
        for source in [name_text, full_text]:
            for line in source.split("\n"):
                clean = re.sub(r"[^a-zA-ZÀ-ÿ\s\-éèêëàâùûîïôœç']+", "", line).strip()
                # Doit commencer par une majuscule et faire >2 chars
                if len(clean) > 2 and clean[0].isupper():
                    # Éliminer les faux positifs (mots trop courts)
                    words = [w for w in clean.split() if len(w) > 1]
                    if words:
                        return " ".join(words)
        return ""

    def _parse_number(self, number_text: str, full_text: str) -> str:
        """
        Extrait le numéro au format X/Y (ex : '4/102', '025/198').
        Gère aussi les formats promos (ex : 'SM-P', 'SWSH-P').
        """
        pattern_std   = r"\b(\d{1,3})\s*/\s*(\d{1,3})\b"
        pattern_promo = r"\b([A-Z]{2,4}[\-\s]?[A-Z0-9]+)\b"

        for text in [number_text, full_text]:
            m = re.search(pattern_std, text)
            if m:
                return f"{m.group(1)}/{m.group(2)}"

        for text in [number_text, full_text]:
            m = re.search(pattern_promo, text)
            if m and len(m.group(1)) > 3:
                return m.group(1)

        return ""

    def _parse_hp(self, hp_text: str) -> Optional[int]:
        """Extrait les PV (ex : '120 PV' → 120)."""
        m = re.search(r"(\d{1,4})\s*(?:PV|HP)", hp_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None
