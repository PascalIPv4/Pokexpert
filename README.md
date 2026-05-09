# 🃏 PokeExpert FR Scan Bot

> Identifiez et cotez vos cartes Pokémon françaises en temps réel, depuis votre navigateur.
> Conçu pour une utilisation en vide-grenier : **scan → état → cote en 5-8 secondes**.

---

## Fonctionnalités

| Feature | Détail |
|---|---|
| 📸 **Scan par photo** | Upload d'une photo de carte depuis mobile ou desktop |
| 🔧 **Redressement de perspective** | Correction automatique de l'angle et de la luminosité (OpenCV) |
| 🔤 **OCR Tesseract** | Extraction du nom et du numéro de série |
| 🤖 **Claude Vision** *(optionnel)* | Identification haute précision via IA si clé Anthropic fournie |
| 🔍 **API Pokémon TCG** | Base de données officielle pour l'identification de la carte |
| 📊 **Cardmarket FR** | Prix Trend et prix moyen (langue française, `idLanguage=2`) |
| 🛒 **eBay FR** | Scraping des 5 dernières ventes réussies + indice de liquidité |
| ⚖️ **Cote pondérée** | 70 % Cardmarket + 30 % eBay, ajustée aux variantes |
| ✏️ **Saisie manuelle** | Fallback si confiance OCR < 80 % |

---

## Installation

### 1. Prérequis système — Tesseract OCR

**macOS :**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu / Debian :**
```bash
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
```

**Windows :**
Téléchargez l'installateur depuis [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
et ajoutez `C:\Program Files\Tesseract-OCR` à votre `PATH`.

Vérification :
```bash
tesseract --version
```

---

### 2. Dépendances Python

```bash
# Cloner / décompresser le projet
cd pokeexpert

# Environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

### 3. Configuration (optionnel mais recommandé)

Copiez `.env.example` en `.env` :
```bash
cp .env.example .env
```

Renseignez vos clés dans `.env` :
```dotenv
# API Pokémon TCG — gratuit sur https://dev.pokemontcg.io/
# Sans clé : 1 000 req/jour. Avec clé : 20 000 req/jour.
POKEMON_TCG_API_KEY=votre_cle_ici

# Anthropic (Claude Vision) — optionnel, améliore significativement la précision
# https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-...
```

---

### 4. Lancement

```bash
streamlit run app.py
```

Ouvrez [http://localhost:8501](http://localhost:8501) dans votre navigateur.

**Sur mobile (vide-grenier) :**
Lancez le serveur sur votre PC/laptop et accédez via l'IP locale :
`http://192.168.x.x:8501`

---

## Architecture du projet

```
pokeexpert/
├── app.py                      ← Interface Streamlit (point d'entrée)
├── config.py                   ← Paramètres globaux, clés API
├── requirements.txt
├── .env.example
├── README.md
└── modules/
    ├── __init__.py
    ├── image_processor.py      ← OpenCV + Tesseract OCR
    ├── card_identifier.py      ← API Pokémon TCG + fuzzy matching + Claude Vision
    └── price_fetcher.py        ← Scraping Cardmarket FR + eBay FR
```

---

## Pipeline technique

```
Photo uploadée
     │
     ▼
ImageProcessor.process()
  ├─ Détection des 4 coins (Canny + contours)
  ├─ Warp perspective → vue orthogonale
  ├─ Amélioration adaptative (contraste, netteté, luminosité)
  └─ OCR par zones (nom, numéro, PV) + nettoyage

     │  OcrResult {parsed_name, parsed_number, ...}
     ▼

CardIdentifier.identify()
  ├─ [Si ANTHROPIC_API_KEY] → Claude Vision → nom + numéro précis
  ├─ Requête API Pokémon TCG (pokemontcg.io)
  ├─ Fuzzy matching (rapidfuzz) sur les résultats
  └─ Score de confiance 0-100 %

     │  CardData + confidence
     ▼
  ≥ 80 % → Étape 2 (modificateurs)
  < 80 % → Demande saisie manuelle

     │  condition (MT/NM/EX/…) + variantes
     ▼

PriceFetcher.get_prices()
  ├─ Cardmarket FR
  │   ├─ URL directe /Singles/{set}/{card}?language=2
  │   ├─ Fallback : moteur de recherche CM
  │   └─ Parse price-guide (Trend, Avg, Low)
  ├─ eBay FR
  │   ├─ Ventes réussies (LH_Sold=1, LH_Complete=1, LH_PrefLoc=1)
  │   └─ Parse 5 dernières ventes + indice liquidité
  └─ Moyenne pondérée (70% CM + 30% eBay)

     │
     ▼
Affichage : Cote Expert + détails + liquidité
```

---

## Calcul de la cote

### Modificateurs état
L'état est transmis aux requêtes Cardmarket et eBay comme filtre de recherche.
Sur Cardmarket, les prix affichés correspondent à l'état sélectionné.

### Modificateurs variantes
Si une variante est sélectionnée mais que Cardmarket ne propose pas de page dédiée,
un multiplicateur est appliqué sur le prix de base :

| Variante | Multiplicateur |
|---|---|
| Holo | ×1.5 |
| Reverse | ×1.3 |
| Première Édition | ×2.5 |
| Stampé | ×1.4 |

> ⚠️ Ces multiplicateurs sont des **estimations statistiques** basées sur le marché FR.
> Les prix réels peuvent varier selon la rareté et la demande.

### Pondération finale
```
Cote = (Prix Cardmarket × 0.70) + (Prix eBay moyen × 0.30)
```
Si une seule source est disponible, elle est utilisée à 100%.

---

## Dépannage

| Problème | Solution |
|---|---|
| `pytesseract.pytesseract.TesseractNotFoundError` | Tesseract n'est pas installé ou pas dans le PATH |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python-headless` |
| OCR retourne du charabia | Prise de vue trop floue ou lumière insuffisante |
| Confiance toujours < 80 % | Utiliser Claude Vision (ajouter `ANTHROPIC_API_KEY`) |
| Cardmarket retourne 403 | Attendre quelques minutes (rate limiting) |
| eBay retourne 0 résultat | La carte est peu commune sur eBay FR — cote CM uniquement |

---

## Limitations connues

- **Scraping** : Cardmarket et eBay peuvent modifier leur HTML à tout moment.
  Si les prix ne s'affichent plus, vérifier les sélecteurs CSS dans `price_fetcher.py`.
- **API TCG** : Les noms de cartes sont en anglais dans la base. La traduction FR
  s'effectue via Cardmarket lors de la récupération des prix.
- **Cartes très rares / promos** : L'API TCG peut ne pas les référencer.
  Utiliser la saisie manuelle avec le numéro de promo.
- **CAPTCHA** : eBay peut présenter un CAPTCHA après de nombreuses requêtes.
  Réduire la fréquence d'utilisation ou utiliser un proxy si besoin.

---

## Roadmap

- [ ] Bot Telegram (`python-telegram-bot`) comme alternative mobile native
- [ ] Cache Redis pour éviter les requêtes répétées sur les mêmes cartes
- [ ] Historique des cotations en base SQLite
- [ ] Reconnaissance par hash d'image (fingerprinting) pour les cartes connues
- [ ] Support des cartes japonaises

---

## Licence

Usage personnel / éducatif. Le scraping de sites tiers doit respecter leurs CGU.
