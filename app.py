"""
app.py — PokeExpert FR Scan Bot
Interface Streamlit pour identifier et coter des cartes Pokémon françaises.

Lancement : streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
import streamlit as st
from PIL import Image

from config import CONDITIONS, CONFIDENCE_THRESHOLD
from modules import CardIdentifier, CardData, ImageProcessor, OcrResult, PriceFetcher

# ─────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PokeExpert FR Scan",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS personnalisé — thème sombre avec accent jaune Pokémon
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@400;600;700&family=Space+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'Kanit', sans-serif; }

    .stApp { background: #0d0d14; color: #f0f0f0; }

    h1 { color: #ffcb05 !important; letter-spacing: 1px; }
    h2, h3 { color: #e0e0e0 !important; }

    .card-badge {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #ffcb05;
        border-radius: 8px;
        padding: 4px 12px;
        font-size: 13px;
        color: #ffcb05;
        margin: 2px;
        font-family: 'Space Mono', monospace;
    }

    .price-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #ffcb05;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 12px 0;
    }
    .price-hero .amount {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffcb05;
        line-height: 1.1;
    }
    .price-hero .label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
    }

    .liquidity-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 13px;
        font-weight: 600;
    }
    .liq-high  { background: #14532d; color: #4ade80; }
    .liq-mid   { background: #713f12; color: #fbbf24; }
    .liq-low   { background: #450a0a; color: #f87171; }

    .sale-item {
        background: #1e1e2e;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 8px 14px;
        margin: 5px 0;
        font-size: 14px;
    }

    .warn-box {
        background: #2d1b00;
        border: 1px solid #f97316;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fdba74;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] { color: #ffcb05 !important; font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"] { color: #aaa !important; }

    .stButton > button {
        background: #ffcb05 !important;
        color: #0d0d14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Kanit', sans-serif !important;
        font-size: 16px !important;
        padding: 10px 28px !important;
    }
    .stButton > button:hover { background: #f0b800 !important; }

    .stSelectSlider > div > div { color: #f0f0f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "card": None,
        "ocr": None,
        "processed_img": None,
        "prices": None,
        "low_confidence": False,
        "step": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ─────────────────────────────────────────────
# Modules (cached pour ne pas les réinstancier à chaque rerun)
# ─────────────────────────────────────────────

@st.cache_resource
def load_modules():
    return ImageProcessor(), CardIdentifier(), PriceFetcher()


processor, identifier, fetcher = load_modules()


# ─────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────

def _rarity_emoji(rarity: str) -> str:
    mapping = {
        "Common": "⚪", "Uncommon": "🟢", "Rare": "🔵",
        "Rare Holo": "💫", "Rare Ultra": "🌟",
        "Rare Secret": "✨", "Amazing Rare": "🌈",
        "Promo": "🎁",
    }
    return mapping.get(rarity, "❓")


def _liquidity_html(count: int) -> str:
    if count >= 10:
        cls, label = "liq-high", f"🟢 {count} ventes/30j — Très liquide"
    elif count >= 4:
        cls, label = "liq-mid", f"🟡 {count} ventes/30j — Liquide"
    else:
        cls, label = "liq-low", f"🔴 {count} ventes/30j — Peu liquide"
    return f'<span class="liquidity-badge {cls}">{label}</span>'


def _display_card_info(card: CardData):
    """Affiche les informations de la carte identifiée."""
    col_img, col_info = st.columns([1, 2])

    with col_img:
        if card.image_url:
            st.image(card.image_url, width=180)
        else:
            st.info("Pas d'image disponible")

    with col_info:
        st.markdown(f"### {card.name_fr or card.name}")
        st.markdown(
            f'<span class="card-badge">{card.set_name}</span>'
            f'<span class="card-badge">#{card.number}</span>'
            f'<span class="card-badge">{_rarity_emoji(card.rarity)} {card.rarity}</span>'
            + (f'<span class="card-badge">❤️ {card.hp} PV</span>' if card.hp else ""),
            unsafe_allow_html=True,
        )
        if card.set_series:
            st.caption(f"Série : {card.set_series}")
        if card.supertype:
            st.caption(f"Type : {card.supertype} — {', '.join(card.subtypes)}")


# ─────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────

# Header
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <span style="font-size:2.5rem">🃏</span>
        <div>
            <h1 style="margin:0;font-size:2rem">PokeExpert FR Scan Bot</h1>
            <p style="margin:0;color:#888;font-size:0.9rem">
                Cotation en temps réel • Cardmarket + eBay FR • Marché français
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────
# ÉTAPE 1 : Upload + Analyse
# ─────────────────────────────────────────────

st.subheader("📸 Étape 1 — Scanner la carte")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Photo de la carte Pokémon française",
        type=["jpg", "jpeg", "png", "webp", "heic"],
        help="📌 Conseil : posez la carte à plat, bonne lumière, évitez le flash",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with col_preview:
            st.image(image, caption="Image uploadée", use_column_width=True)

        btn_scan = st.button("🔍 Analyser la carte", type="primary", use_container_width=True)

        if btn_scan:
            t0 = time.time()
            with st.spinner("⚙️ Traitement de l'image…"):
                img_array = np.array(image)
                processed_img, ocr_result = processor.process(img_array)
                st.session_state.processed_img = processed_img
                st.session_state.ocr = ocr_result

            with st.spinner("🔎 Identification de la carte…"):
                card, confidence = identifier.identify(ocr_result, processed_img)

            elapsed = time.time() - t0

            if card is None or confidence < CONFIDENCE_THRESHOLD:
                st.session_state.low_confidence = True
                st.session_state.card = card
                st.markdown(
                    f'<div class="warn-box">⚠️ Confiance insuffisante '
                    f'({confidence:.0f}% < {CONFIDENCE_THRESHOLD}%). '
                    f'OCR extrait : <b>{ocr_result.parsed_name!r}</b> / '
                    f'<b>{ocr_result.parsed_number!r}</b>.<br>'
                    f'Veuillez utiliser la saisie manuelle ci-dessous ou reprendre la photo.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.session_state.card = card
                st.session_state.low_confidence = False
                st.session_state.prices = None
                st.session_state.step = 2
                st.success(
                    f"✅ Carte identifiée avec **{confidence:.0f}%** de confiance "
                    f"— en {elapsed:.1f}s"
                )
                st.rerun()

# ─────────────────────────────────────────────
# Saisie manuelle (confiance faible)
# ─────────────────────────────────────────────

if st.session_state.low_confidence or not uploaded:
    with st.expander("✏️ Saisie manuelle (si scan échoue)", expanded=st.session_state.low_confidence):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            manual_name = st.text_input(
                "Nom du Pokémon (FR ou EN)",
                value=getattr(st.session_state.ocr, "parsed_name", "") if st.session_state.ocr else "",
            )
        with m_col2:
            manual_number = st.text_input(
                "Numéro de série (ex : 4/102)",
                value=getattr(st.session_state.ocr, "parsed_number", "") if st.session_state.ocr else "",
            )

        if st.button("🔎 Identifier manuellement", use_container_width=True):
            with st.spinner("Recherche en cours…"):
                card, confidence = identifier.identify_manual(manual_name, manual_number)
            if card:
                st.session_state.card = card
                st.session_state.low_confidence = False
                st.session_state.prices = None
                st.session_state.step = 2
                st.success(f"✅ Trouvé : **{card.name}** ({confidence:.0f}%)")
                st.rerun()
            else:
                st.error("❌ Carte introuvable. Vérifiez le nom ou le numéro.")

# ─────────────────────────────────────────────
# ÉTAPE 2 : Modificateurs & Cotation
# ─────────────────────────────────────────────

if st.session_state.card:
    card: CardData = st.session_state.card
    st.divider()
    st.subheader("🎴 Carte identifiée")
    _display_card_info(card)

    st.divider()
    st.subheader("⚙️ Étape 2 — Modificateurs Expert")

    mod_col1, mod_col2 = st.columns([1, 1])

    with mod_col1:
        condition = st.select_slider(
            "État de la carte (échelle MKM)",
            options=list(CONDITIONS.keys()),
            value="NM",
            format_func=lambda k: f"{k} — {CONDITIONS[k]['label']}",
        )
        # Afficher la couleur de l'état
        cond_color = CONDITIONS[condition]["color"]
        st.markdown(
            f'<div style="width:100%;height:6px;background:{cond_color};'
            f'border-radius:3px;margin-top:-8px"></div>',
            unsafe_allow_html=True,
        )

    with mod_col2:
        st.write("**Variantes :**")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            is_holo      = st.checkbox("✨ Holo")
            is_reverse   = st.checkbox("🔄 Reverse")
        with v_col2:
            is_first_ed  = st.checkbox("1️⃣ 1ère Édition")
            is_stamped   = st.checkbox("🔖 Stampé")

    variants = {
        "holo":          is_holo,
        "reverse":       is_reverse,
        "first_edition": is_first_ed,
        "stamped":       is_stamped,
    }

    if st.button("💶 Obtenir la cote en temps réel", type="primary", use_container_width=True):
        t0 = time.time()
        with st.spinner("Cardmarket + eBay FR en cours…"):
            prices = fetcher.get_prices(card, condition, variants)
            st.session_state.prices = prices
        elapsed = time.time() - t0
        st.caption(f"Récupération des prix : {elapsed:.1f}s")
        st.rerun()

# ─────────────────────────────────────────────
# ÉTAPE 3 : Résultats
# ─────────────────────────────────────────────

if st.session_state.prices:
    prices = st.session_state.prices
    cm    = prices.cm
    ebay  = prices.ebay

    st.divider()
    st.subheader("💰 Estimation Expert")

    # Hero price
    if prices.weighted_avg:
        variants_label = (
            " + ".join(v.replace("_", " ").title() for v in prices.variants_applied)
            or "Standard"
        )
        source_label = "70% Cardmarket · 30% eBay FR"
        if not cm.trend_price and not cm.avg_sell_price:
            source_label = "100% eBay FR (Cardmarket indisponible)"
        elif not ebay.avg_price:
            source_label = "100% Cardmarket (aucune vente eBay)"

        st.markdown(
            f"""
            <div class="price-hero">
                <div class="label">COTE EXPERT — {prices.condition} · {variants_label}</div>
                <div class="amount">{prices.weighted_avg:.2f} €</div>
                <div class="label" style="margin-top:8px">{source_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Impossible de calculer une cote. Aucune donnée trouvée.")

    # Metrics détaillées
    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.metric(
            "📊 Trend Cardmarket",
            f"{cm.trend_price:.2f} €" if cm.trend_price else "N/D",
            help="Prix tendance (moyenne des ventes récentes sur Cardmarket)",
        )
        if cm.low_price:
            st.caption(f"Prix le plus bas : {cm.low_price:.2f} €")
        if cm.variant_multiplier_applied:
            st.caption("⚠️ Multiplicateur variante appliqué")

    with detail_col2:
        st.metric(
            "🛒 Moyenne eBay FR",
            f"{ebay.avg_price:.2f} €" if ebay.avg_price else "N/D",
            help="Moyenne des ventes réussies récentes sur eBay.fr",
        )
        if ebay.min_price and ebay.max_price:
            st.caption(f"Fourchette : {ebay.min_price:.2f} € → {ebay.max_price:.2f} €")

    with detail_col3:
        st.metric(
            "📈 Liquidité (30j)",
            f"~{ebay.sold_30d} ventes",
            help="Estimation des ventes réussies sur eBay FR ces 30 derniers jours",
        )
        st.markdown(_liquidity_html(ebay.sold_30d), unsafe_allow_html=True)

    # Dernières ventes eBay
    if ebay.sales:
        st.divider()
        st.subheader("🛒 5 dernières ventes eBay FR")
        for sale in ebay.sales:
            link = f'<a href="{sale.url}" target="_blank" style="color:#60a5fa">🔗</a>' if sale.url else ""
            st.markdown(
                f'<div class="sale-item">'
                f'<b style="color:#ffcb05">{sale.price:.2f} €</b> &nbsp;'
                f'<span style="color:#aaa;font-size:12px">{sale.date}</span> {link}<br>'
                f'<span style="font-size:13px">{sale.title}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif ebay.error:
        st.warning(f"⚠️ eBay : {ebay.error}")
    else:
        st.info("Aucune vente eBay récente trouvée pour cette carte.")

    if cm.error:
        st.warning(f"⚠️ Cardmarket : {cm.error}")

    if not prices.has_recent_sales:
        st.markdown(
            '<div class="warn-box">📉 Peu ou pas de ventes récentes. '
            "La cote est indicative et peut ne pas refléter le marché actuel.</div>",
            unsafe_allow_html=True,
        )

    # Bouton reset
    st.divider()
    if st.button("🔄 Analyser une autre carte"):
        for key in ("card", "ocr", "processed_img", "prices", "low_confidence", "step"):
            st.session_state[key] = None if key != "low_confidence" else False
        st.session_state["step"] = 1
        st.rerun()
