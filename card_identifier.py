"""
modules/card_identifier.py
Identification d'une carte Pokémon via l'API officielle Pokémon TCG.
Utilise la correspondance floue (fuzzywuzzy) pour gérer les erreurs OCR.
Supporte également la reconnaissance via Claude Vision (optionnel).
"""

import re
import json
import base64
import requests
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
import io

try:
    from rapidfuzz import fuzz, process as fuzz_process
    _FUZZY_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process as fuzz_process
        _FUZZY_AVAILABLE = True
    except ImportError:
        _FUZZY_AVAILABLE = False

from config import POKEMON_TCG_API_KEY, ANTHROPIC_API_KEY, CONFIDENCE_THRESHOLD, REQUEST_TIMEOUT


@dataclass
class CardData:
    """Représentation normalisée d'une carte Pokémon identifiée."""
    id: str = ""
    name: str = ""
    name_fr: str = ""
    number: str = ""
    set_id: str = ""
    set_name: str = ""
    set_series: str = ""
    rarity: str = ""
    image_url: str = ""
    image_small: str = ""
    national_dex: list = field(default_factory=list)
    supertype: str = ""
    subtypes: list = field(default_factory=list)
    hp: str = ""
    # Données brutes pour usage avancé
    raw: dict = field(default_factory=dict)

    def to_cardmarket_slug(self) -> tuple[str, str]:
        """Génère les slugs Cardmarket (set, card) depuis les données."""
        set_slug = _slugify(self.set_name)
        card_slug = _slugify(self.name_fr or self.name)
        return set_slug, card_slug

    def to_ebay_query(self, variants: dict) -> str:
        """Génère une requête eBay optimisée."""
        parts = [f"Pokémon {self.name_fr or self.name}"]
        if self.number:
            parts.append(self.number)
        if variants.get("holo"):
            parts.append("Holo")
        if variants.get("reverse"):
            parts.append("Reverse")
        if variants.get("first_edition"):
            parts.append("1ère édition")
        if variants.get("stamped"):
            parts.append("Stamp")
        parts.append("FR")
        return " ".join(parts)


class CardIdentifier:
    """
    Identification de cartes Pokémon via :
        1. API Pokémon TCG (pokemontcg.io) — principale
        2. Claude Vision (Anthropic API) — optionnel, meilleure précision
        3. Correspondance floue sur le texte OCR

    Usage :
        identifier = CardIdentifier()
        card, confidence = identifier.identify(ocr_result, img_array)
    """

    TCG_API_BASE = "https://api.pokemontcg.io/v2"
    ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"

    def __init__(self):
        self._tcg_headers = {}
        if POKEMON_TCG_API_KEY:
            self._tcg_headers["X-Api-Key"] = POKEMON_TCG_API_KEY

    # ─────────────────────────────────────────────
    # Point d'entrée principal
    # ─────────────────────────────────────────────

    def identify(
        self,
        ocr_result,
        img_array: Optional[np.ndarray] = None,
    ) -> tuple[Optional[CardData], float]:
        """
        Identifie la carte à partir de l'OCR + image (optionnelle).

        Strategy :
            A. Si Claude Vision disponible → on l'utilise en priorité
            B. Sinon → API TCG + fuzzy matching sur les résultats OCR

        Returns:
            (CardData | None, confidence_score 0-100)
        """
        name = getattr(ocr_result, "parsed_name", "") or ""
        number = getattr(ocr_result, "parsed_number", "") or ""

        # --- Stratégie A : Claude Vision (précision maximale) ---
        if ANTHROPIC_API_KEY and img_array is not None:
            vision_result = self._identify_via_claude_vision(img_array)
            if vision_result:
                name = vision_result.get("name", name) or name
                number = vision_result.get("number", number) or number
                # On booste la confiance si Claude a répondu
                vision_confidence = vision_result.get("confidence", 0)
                if vision_confidence >= CONFIDENCE_THRESHOLD:
                    cards = self._search_api(name=name, number=number)
                    if cards:
                        card, conf = self._rank_cards(cards, name, number)
                        return card, max(conf, vision_confidence)

        # --- Stratégie B : API TCG + fuzzy ---
        if not name and not number:
            return None, 0.0

        # Tentatives progressives
        for search_kwargs in [
            {"name": name, "number": number},
            {"name": name},
            {"number": number},
        ]:
            if not any(search_kwargs.values()):
                continue
            cards = self._search_api(**search_kwargs)
            if cards:
                card, confidence = self._rank_cards(cards, name, number)
                return card, confidence

        return None, 0.0

    def identify_manual(self, name: str, number: str) -> tuple[Optional[CardData], float]:
        """Identification manuelle par nom + numéro (saisie utilisateur)."""
        cards = self._search_api(name=name.strip(), number=number.strip())
        if not cards:
            cards = self._search_api(name=name.strip())
        if not cards:
            return None, 0.0
        return self._rank_cards(cards, name, number)

    # ─────────────────────────────────────────────
    # Claude Vision (optionnel)
    # ─────────────────────────────────────────────

    def _identify_via_claude_vision(self, img_array: np.ndarray) -> Optional[dict]:
        """
        Envoie l'image à Claude Vision pour obtenir :
        name, number, set, rarity, language, confidence.
        """
        try:
            # Convertir l'image en base64
            pil = Image.fromarray(img_array)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Tu es un expert en cartes Pokémon. Analyse cette carte et réponds "
                                    "UNIQUEMENT en JSON valide (sans markdown) avec ces champs :\n"
                                    "{\n"
                                    '  "name": "Nom du Pokémon en français",\n'
                                    '  "number": "Numéro (ex: 4/102)",\n'
                                    '  "set": "Nom de l\'extension",\n'
                                    '  "rarity": "Rareté",\n'
                                    '  "language": "fr ou autre",\n'
                                    '  "is_holo": true/false,\n'
                                    '  "is_reverse": true/false,\n'
                                    '  "is_first_edition": true/false,\n'
                                    '  "confidence": 0 à 100\n'
                                    "}\n"
                                    "Si tu n'es pas sûr, mets confidence < 80."
                                ),
                            },
                        ],
                    }
                ],
            }

            resp = requests.post(
                f"{self.ANTHROPIC_API_BASE}/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            # Nettoyer les backticks éventuels
            content = re.sub(r"```json|```", "", content).strip()
            return json.loads(content)

        except Exception as e:
            print(f"[Claude Vision] Erreur : {e}")
            return None

    # ─────────────────────────────────────────────
    # API Pokémon TCG
    # ─────────────────────────────────────────────

    def _search_api(self, name: str = "", number: str = "") -> list[dict]:
        """Requête l'API pokemontcg.io avec les filtres disponibles."""
        query_parts = []

        if name:
            normalized = name.strip().title()
            # Translittérer les caractères spéciaux courants en OCR
            normalized = normalized.replace("é", "e")
            query_parts.append(f'name:"{normalized}"')

        if number:
            num_only = number.split("/")[0].lstrip("0") or "0"
            query_parts.append(f"number:{num_only}")

        if not query_parts:
            return []

        query = " ".join(query_parts)
        try:
            resp = requests.get(
                f"{self.TCG_API_BASE}/cards",
                params={
                    "q": query,
                    "pageSize": 20,
                    "select": (
                        "id,name,number,set,rarity,images,"
                        "nationalPokedexNumbers,supertype,subtypes,hp"
                    ),
                },
                headers=self._tcg_headers,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            print(f"[TCG API] Erreur : {e}")
            return []

    # ─────────────────────────────────────────────
    # Scoring / Ranking
    # ─────────────────────────────────────────────

    def _rank_cards(
        self, cards: list[dict], ocr_name: str, ocr_number: str
    ) -> tuple[Optional[CardData], float]:
        """
        Attribue un score à chaque carte et retourne la meilleure.

        Pondération :
            - Similarité du nom (fuzzy)      : 55 pts max
            - Correspondance du numéro exact : 40 pts
            - Bonus si même set trouvé        :  5 pts
        """
        best_card = None
        best_score = 0.0

        for card in cards:
            score = 0.0

            # Score nom
            if ocr_name and _FUZZY_AVAILABLE:
                name_score = fuzz.ratio(
                    ocr_name.lower(), card.get("name", "").lower()
                )
                score += name_score * 0.55
            elif ocr_name:
                # Fallback basique si rapidfuzz/fuzzywuzzy absent
                if ocr_name.lower() in card.get("name", "").lower():
                    score += 40.0

            # Score numéro
            if ocr_number:
                ocr_num = ocr_number.split("/")[0].lstrip("0") or "0"
                card_num = str(card.get("number", "")).lstrip("0") or "0"
                if ocr_num == card_num:
                    score += 40.0
                elif card_num.startswith(ocr_num):
                    score += 20.0

            if score > best_score:
                best_score = score
                best_card = card

        if not best_card:
            # Fallback : premier résultat avec confiance basse
            best_card = cards[0]
            best_score = 40.0

        return self._format_card(best_card), min(best_score, 100.0)

    def _format_card(self, raw: dict) -> CardData:
        """Convertit la réponse brute de l'API en CardData structuré."""
        set_data = raw.get("set", {})
        images = raw.get("images", {})

        return CardData(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            name_fr=raw.get("name", ""),  # L'API TCG est en anglais ; la FR viendra du lookup CM
            number=raw.get("number", ""),
            set_id=set_data.get("id", ""),
            set_name=set_data.get("name", ""),
            set_series=set_data.get("series", ""),
            rarity=raw.get("rarity", ""),
            image_url=images.get("large", images.get("small", "")),
            image_small=images.get("small", ""),
            national_dex=raw.get("nationalPokedexNumbers", []),
            supertype=raw.get("supertype", ""),
            subtypes=raw.get("subtypes", []),
            hp=raw.get("hp", ""),
            raw=raw,
        )


# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convertit un texte en slug compatible Cardmarket (kebab-case)."""
    # Translittérations communes
    replacements = {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "ù": "u", "û": "u", "ü": "u",
        "î": "i", "ï": "i",
        "ô": "o", "œ": "oe",
        "ç": "c", "'": "-", "'": "-",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)

    # Supprimer les caractères non-alphanumériques (sauf tirets et espaces)
    text = re.sub(r"[^\w\s\-]", "", text.lower())
    # Remplacer espaces par tirets
    text = re.sub(r"\s+", "-", text.strip())
    # Supprimer tirets multiples
    text = re.sub(r"-+", "-", text)
    return text
