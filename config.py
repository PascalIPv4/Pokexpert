"""
PokeExpert FR Scan Bot — Configuration
Renseignez vos clés API ici (ou via variables d'environnement).
"""

import os

# === API KEYS ===
# Pokémon TCG API (gratuit, mais limité sans clé : 1000 req/jour)
# Obtenez votre clé sur https://dev.pokemontcg.io/
POKEMON_TCG_API_KEY = os.getenv("POKEMON_TCG_API_KEY", "")

# Anthropic API (optionnel — active la reconnaissance d'image via Claude Vision)
# Clé récupérable sur https://console.anthropic.com/
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# === PARAMÈTRES DE L'APPLICATION ===

# Seuil de confiance minimum (%) pour valider une identification automatique
CONFIDENCE_THRESHOLD = 80

# Pondération du calcul de prix (doit totaliser 100)
CM_WEIGHT = 0.70       # Cardmarket
EBAY_WEIGHT = 0.30     # eBay FR

# Timeout des requêtes web (secondes)
REQUEST_TIMEOUT = 10

# Multiplicateurs de prix par variante
VARIANT_MULTIPLIERS = {
    "holo": 1.5,
    "reverse": 1.3,
    "first_edition": 2.5,
    "stamped": 1.4,
}

# === ÉTATS CARDMARKET (MKM) ===
CONDITIONS = {
    "MT": {"label": "Mint", "cm_id": 1, "color": "#4ade80"},
    "NM": {"label": "Near Mint", "cm_id": 2, "color": "#86efac"},
    "EX": {"label": "Excellent", "cm_id": 3, "color": "#fbbf24"},
    "GD": {"label": "Good", "cm_id": 4, "color": "#f97316"},
    "LP": {"label": "Light Played", "cm_id": 5, "color": "#fb923c"},
    "PL": {"label": "Played", "cm_id": 6, "color": "#ef4444"},
    "PO": {"label": "Poor", "cm_id": 7, "color": "#7f1d1d"},
}

# === USER-AGENT pour le scraping ===
SCRAPING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.fr/",
}
