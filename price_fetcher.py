"""
modules/price_fetcher.py
Récupération des prix en temps réel depuis :
    - Cardmarket FR (idLanguage=2 → Français)
    - eBay FR (ventes réussies uniquement, localisation France)

Calcul d'une cote pondérée : 70% Cardmarket + 30% eBay.
"""

import re
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import (
    CM_WEIGHT,
    EBAY_WEIGHT,
    SCRAPING_HEADERS,
    REQUEST_TIMEOUT,
    VARIANT_MULTIPLIERS,
)


# ─────────────────────────────────────────────
# Dataclasses de résultat
# ─────────────────────────────────────────────

@dataclass
class EbaySale:
    title: str
    price: float
    date: str
    url: str = ""


@dataclass
class EbayResult:
    sales: list[EbaySale] = field(default_factory=list)
    avg_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    sold_30d: int = 0
    total_found: int = 0
    error: str = ""


@dataclass
class CardmarketResult:
    trend_price: Optional[float] = None
    avg_sell_price: Optional[float] = None
    low_price: Optional[float] = None
    avg_30d: Optional[float] = None
    available_items: int = 0
    variant_multiplier_applied: bool = False
    url: str = ""
    error: str = ""


@dataclass
class PriceResult:
    weighted_avg: Optional[float] = None
    cm: CardmarketResult = field(default_factory=CardmarketResult)
    ebay: EbayResult = field(default_factory=EbayResult)
    has_recent_sales: bool = False
    condition: str = "NM"
    variants_applied: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# PriceFetcher
# ─────────────────────────────────────────────

class PriceFetcher:
    """
    Orchestre la récupération des prix depuis Cardmarket et eBay FR.

    Usage :
        fetcher = PriceFetcher()
        result = fetcher.get_prices(card_data, condition="NM", variants={...})
    """

    CM_BASE = "https://www.cardmarket.com"
    EBAY_BASE = "https://www.ebay.fr"

    # Map état → filtre eBay (approximatif)
    _EBAY_CONDITION_KEYWORDS = {
        "MT": "mint",
        "NM": "near mint",
        "EX": "excellent",
        "GD": "good",
        "LP": "light played",
        "PL": "played",
        "PO": "poor",
    }

    def get_prices(
        self,
        card,  # CardData
        condition: str = "NM",
        variants: Optional[dict] = None,
    ) -> PriceResult:
        """
        Point d'entrée principal.

        Args:
            card      : CardData (depuis card_identifier)
            condition : Code état MKM (MT, NM, EX, GD, LP, PL, PO)
            variants  : dict {holo, reverse, first_edition, stamped} → bool

        Returns:
            PriceResult complet
        """
        if variants is None:
            variants = {}

        result = PriceResult(condition=condition)

        # Collecter les variantes actives
        result.variants_applied = [k for k, v in variants.items() if v]

        # 1. Cardmarket
        cm = self._fetch_cardmarket(card, condition, variants)
        result.cm = cm

        # 2. eBay FR
        ebay = self._fetch_ebay(card, condition, variants)
        result.ebay = ebay

        # 3. Cote pondérée
        cm_price  = cm.trend_price or cm.avg_sell_price
        ebay_price = ebay.avg_price

        if cm_price and ebay_price:
            result.weighted_avg = round(
                cm_price * CM_WEIGHT + ebay_price * EBAY_WEIGHT, 2
            )
            result.has_recent_sales = True
        elif cm_price:
            result.weighted_avg = cm_price
            result.has_recent_sales = ebay.total_found > 0
        elif ebay_price:
            result.weighted_avg = ebay_price
            result.has_recent_sales = True

        return result

    # ─────────────────────────────────────────────
    # Cardmarket
    # ─────────────────────────────────────────────

    def _fetch_cardmarket(
        self, card, condition: str, variants: dict
    ) -> CardmarketResult:
        """
        Tente plusieurs stratégies pour récupérer le prix Cardmarket :
            1. URL directe du produit (set + card slugs)
            2. Recherche textuelle fallback
        """
        set_slug, card_slug = card.to_cardmarket_slug()

        # Stratégie 1 : URL directe
        direct_url = (
            f"{self.CM_BASE}/fr/Pokemon/Products/Singles/"
            f"{set_slug}/{card_slug}?language=2"
        )
        cm = self._scrape_cardmarket_page(direct_url, condition, variants)
        if cm.trend_price or cm.avg_sell_price:
            cm.url = direct_url
            return cm

        # Stratégie 2 : recherche via le moteur CM
        search_url = (
            f"{self.CM_BASE}/fr/Pokemon/Products/Search"
            f"?searchString={urllib.parse.quote(card.name_fr or card.name)}"
            f"&idLanguage=2&number={urllib.parse.quote(card.number)}"
        )
        cm2 = self._scrape_cardmarket_search(search_url, card, condition, variants)
        if cm2.trend_price or cm2.avg_sell_price:
            cm2.url = search_url
            return cm2

        cm.error = "Produit non trouvé sur Cardmarket"
        return cm

    def _scrape_cardmarket_page(
        self, url: str, condition: str, variants: dict
    ) -> CardmarketResult:
        """Parse une page produit Cardmarket."""
        result = CardmarketResult()
        try:
            resp = requests.get(
                url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}"
                return result

            soup = BeautifulSoup(resp.text, "html.parser")

            # --- Price Guide (tableau résumé) ---
            # Cardmarket affiche un bloc dl avec dt/dd
            price_guide = soup.find("dl", class_=re.compile(r"price", re.I))
            if not price_guide:
                # Essayer une autre structure (CM change parfois)
                price_guide = soup.find("div", class_=re.compile(r"price-guide|priceGuide", re.I))

            if price_guide:
                labels = price_guide.find_all("dt")
                values = price_guide.find_all("dd")
                for dt, dd in zip(labels, values):
                    label = dt.get_text(strip=True).lower()
                    raw_val = dd.get_text(strip=True)
                    price = _parse_price(raw_val)
                    if price is None:
                        continue
                    if any(k in label for k in ("tendance", "trend")):
                        result.trend_price = price
                    elif any(k in label for k in ("moyen", "average", "avg")):
                        result.avg_sell_price = price
                    elif any(k in label for k in ("bas", "low", "mini")):
                        result.low_price = price
                    elif "30" in label:
                        result.avg_30d = price

            # --- Nombre d'articles disponibles ---
            avail_el = soup.find("span", class_=re.compile(r"article.*count|available", re.I))
            if avail_el:
                m = re.search(r"\d+", avail_el.get_text())
                if m:
                    result.available_items = int(m.group())

            # --- Appliquer multiplicateur variante si nécessaire ---
            base_price = result.trend_price or result.avg_sell_price
            if base_price:
                multiplier = _compute_variant_multiplier(variants)
                if multiplier > 1.0:
                    result.trend_price = (
                        round(result.trend_price * multiplier, 2)
                        if result.trend_price
                        else None
                    )
                    result.avg_sell_price = (
                        round(result.avg_sell_price * multiplier, 2)
                        if result.avg_sell_price
                        else None
                    )
                    result.variant_multiplier_applied = True

        except requests.RequestException as e:
            result.error = str(e)
        return result

    def _scrape_cardmarket_search(
        self, url: str, card, condition: str, variants: dict
    ) -> CardmarketResult:
        """Recherche dans le moteur CM et suit le premier résultat pertinent."""
        result = CardmarketResult()
        try:
            resp = requests.get(
                url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT
            )
            soup = BeautifulSoup(resp.text, "html.parser")

            # Chercher le premier lien de produit
            product_link = soup.find("a", href=re.compile(r"/fr/Pokemon/Products/Singles/"))
            if product_link:
                product_url = self.CM_BASE + product_link["href"]
                if "language=2" not in product_url:
                    product_url += "?language=2"
                return self._scrape_cardmarket_page(product_url, condition, variants)

        except Exception as e:
            result.error = str(e)
        return result

    # ─────────────────────────────────────────────
    # eBay FR
    # ─────────────────────────────────────────────

    def _fetch_ebay(self, card, condition: str, variants: dict) -> EbayResult:
        """Scrape les ventes réussies sur eBay FR."""
        query = card.to_ebay_query(variants)

        # Ajouter le mot-clé état si utile
        cond_kw = self._EBAY_CONDITION_KEYWORDS.get(condition, "")

        params = {
            "_nkw": query,
            "LH_Sold": "1",       # Annonces terminées (ventes réussies)
            "LH_Complete": "1",
            "_sacat": "0",
            "LH_PrefLoc": "1",    # France uniquement
            "_sop": "13",          # Tri : Plus récent en premier
            "_ipg": "25",          # 25 résultats par page
        }

        url = f"{self.EBAY_BASE}/sch/i.html"
        try:
            resp = requests.get(
                url,
                params=params,
                headers=SCRAPING_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return EbayResult(error=f"HTTP {resp.status_code}")

            return self._parse_ebay_page(resp.text)

        except requests.RequestException as e:
            return EbayResult(error=str(e))

    def _parse_ebay_page(self, html: str) -> EbayResult:
        """Parse la page de résultats eBay."""
        soup = BeautifulSoup(html, "html.parser")
        sales: list[EbaySale] = []

        # eBay peut utiliser plusieurs classes selon la région/version
        items = soup.find_all("li", class_=re.compile(r"s-item"))

        for item in items:
            # Ignorer les résultats fantômes / publicités
            title_el = (
                item.find("div", class_="s-item__title")
                or item.find("span", class_="s-item__title")
            )
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if "Shop on eBay" in title or not title:
                continue

            # Prix
            price_el = item.find("span", class_="s-item__price")
            price_text = price_el.get_text(strip=True) if price_el else ""
            price = _parse_price(price_text)
            if price is None or price <= 0:
                continue

            # Date de vente
            date_el = (
                item.find("span", class_=re.compile(r"s-item__ended|POSITIVE", re.I))
                or item.find("div", class_=re.compile(r"s-item__title--tag"))
            )
            date_text = date_el.get_text(strip=True) if date_el else "N/C"

            # URL de l'annonce
            link_el = item.find("a", class_="s-item__link")
            item_url = link_el["href"] if link_el and link_el.get("href") else ""

            sales.append(
                EbaySale(
                    title=title[:120],
                    price=price,
                    date=date_text,
                    url=item_url,
                )
            )

        if not sales:
            return EbayResult(total_found=0)

        prices = [s.price for s in sales]

        # Heuristique : on estime les ventes des 30 derniers jours à partir
        # du nombre total (eBay affiche par défaut les 90 derniers jours)
        estimated_30d = max(1, len(sales) // 3)

        return EbayResult(
            sales=sales[:5],       # On garde les 5 plus récentes pour l'affichage
            avg_price=round(sum(prices) / len(prices), 2),
            min_price=min(prices),
            max_price=max(prices),
            sold_30d=estimated_30d,
            total_found=len(sales),
        )


# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def _parse_price(text: str) -> Optional[float]:
    """
    Extrait un float depuis des chaînes comme :
        '12,50 €', '€ 12.50', '12 EUR', '2 offres à 12,00 EUR'
    """
    if not text:
        return None
    # Supprimer symboles monétaires et espaces insécables
    cleaned = re.sub(r"[€$£\xa0\s]", "", text)
    # Normaliser séparateur décimal
    cleaned = cleaned.replace(",", ".")
    # Extraire le premier nombre
    m = re.search(r"(\d+\.?\d*)", cleaned)
    if m:
        try:
            val = float(m.group(1))
            return val if val > 0 else None
        except ValueError:
            pass
    return None


def _compute_variant_multiplier(variants: dict) -> float:
    """
    Calcule le multiplicateur de prix global pour les variantes actives.
    On prend le max (pas de cumul pour éviter les surestimations).
    """
    multiplier = 1.0
    for variant, active in variants.items():
        if active:
            m = VARIANT_MULTIPLIERS.get(variant, 1.0)
            multiplier = max(multiplier, m)
    return multiplier
