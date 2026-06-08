"""
risk_arb_app.py — Interactive Risk Arbitrage Pricer
====================================================
Lancer: streamlit run risk_arb_app.py

Onglets:
  1. 📊 Dashboard    — tous les deals du M&A Monitor, live prices, spreads, signaux
  2. 🔬 Deal Pricer  — analyse détaillée d'un deal avec tous les modèles
  3. 💼 Portefeuille — vue agrégée, VaR, Sharpe, corrélation
  4. 🧠 Modèles      — documentation des modèles utilisés
"""

import streamlit as st
import datetime
import math
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from models import (
    DealFeatures, MasterPricer, MLProbModel,
    OptionsImpliedProb, HazardModel, RegulatoryScorer,
    KellySizer, GreeksCalc, PortfolioOptimizer
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Arb Pricer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0e1117; }
  .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
  .metric-card {
    background: #1e2130; border-radius: 10px; padding: 16px 20px;
    border-left: 4px solid #4c78a8; margin-bottom: 8px;
  }
  .signal-strong { color: #00d4aa; font-weight: 800; font-size: 18px; }
  .signal-buy    { color: #54c768; font-weight: 700; }
  .signal-hold   { color: #f5a623; font-weight: 700; }
  .signal-pass   { color: #e05c5c; font-weight: 700; }
  .model-box {
    background: #1a1e2e; border-radius: 8px; padding: 16px;
    border: 1px solid #2d3250; margin-bottom: 12px;
  }
  div[data-testid="stMetricValue"] { font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

MONITOR_PATH = "1. M&A Monitor (North America) - May 20, 2026.xlsx"

@st.cache_data(ttl=300)
def load_monitor(path: str) -> pd.DataFrame:
    """Charge le M&A Monitor et normalise les colonnes."""
    try:
        wb = pd.read_excel(path, sheet_name="SUMMARY", header=None)
    except Exception:
        return pd.DataFrame()

    # Trouver la ligne d'en-tête (chercher "TARGET")
    header_row = None
    for i, row in wb.iterrows():
        if any("TARGET" in str(v).upper() for v in row.values if v):
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name="SUMMARY", header=header_row)

    # Dédupliquer les colonnes AVANT tout traitement
    cols = [str(c).strip().upper() for c in df.columns]
    seen = {}
    deduped = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)
    df.columns = deduped
    df = df.dropna(how="all")

    # Mapper les colonnes utiles (première occurrence de chaque type)
    rename_map = {}
    for c in df.columns:
        cu = c.upper()
        if "TICKER" not in rename_map.values() and "TARGET" in cu and "DEAL" not in cu:
            rename_map[c] = "TICKER"
        elif "ACQUIRER" not in rename_map.values() and ("ACQUIRER" in cu or "ACQUIROR" in cu):
            rename_map[c] = "ACQUIRER"
        elif "DEAL_TYPE" not in rename_map.values() and ("FRIENDLY" in cu or "HOSTILE" in cu):
            rename_map[c] = "DEAL_TYPE"
        elif "ACQ_TYPE" not in rename_map.values() and ("STRAT" in cu or "FINAN" in cu):
            rename_map[c] = "ACQ_TYPE"
        elif "JURISDICTION" not in rename_map.values() and "JURIS" in cu:
            rename_map[c] = "JURISDICTION"
        elif "ANN_DATE" not in rename_map.values() and "ANN" in cu and "DATE" in cu:
            rename_map[c] = "ANN_DATE"
        elif "EXP_CLOSE" not in rename_map.values() and "EXP" in cu and ("COMP" in cu or "CLOS" in cu or "DATE" in cu):
            rename_map[c] = "EXP_CLOSE"
        elif "DAYS_REM" not in rename_map.values() and "DAYS" in cu and "COMP" in cu:
            rename_map[c] = "DAYS_REM"
        elif "CONSIDERATION" not in rename_map.values() and ("CONSID" in cu or ("DESC" in cu and "DEAL" not in cu)):
            rename_map[c] = "CONSIDERATION"
        elif "CURRENCY" not in rename_map.values() and ("CCY" in cu or "CURRENCY" in cu):
            rename_map[c] = "CURRENCY"
        elif "SPREAD_PCT" not in rename_map.values() and "CASH" in cu and "PREM" in cu:
            rename_map[c] = "SPREAD_PCT"

    df = df.rename(columns=rename_map)

    # Fallback: la 3ème colonne est souvent le ticker
    if "TICKER" not in df.columns and df.shape[1] > 2:
        df = df.rename(columns={list(df.columns)[2]: "TICKER"})

    # Nettoyer et filtrer les vrais tickers
    if "TICKER" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df[df["TICKER"].notna()]
    df = df[~df["TICKER"].astype(str).str.contains("TARGET|#NAME|None|nan", na=False)]
    df["TICKER"] = df["TICKER"].astype(str).str.strip().str.upper()
    df = df[df["TICKER"].str.len().between(1, 10)]

    return df.reset_index(drop=True)


@st.cache_data(ttl=60)
def fetch_price(ticker: str) -> float:
    """Fetch prix live."""
    try:
        stk = yf.Ticker(ticker)
        hist = stk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        info = stk.info
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", ""),
            "market_cap": info.get("marketCap", 0),
            "beta": info.get("beta", 1.0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_history(ticker: str, period: str = "3mo") -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return pd.DataFrame()


def signal_color(sig: str) -> str:
    return {"STRONG BUY": "#00d4aa", "BUY": "#54c768",
            "HOLD": "#f5a623", "PASS": "#e05c5c"}.get(sig, "#888")


def signal_badge(sig: str) -> str:
    icon = {"STRONG BUY": "🟢", "BUY": "🟩", "HOLD": "🟡", "PASS": "🔴"}.get(sig, "⚪")
    return f"{icon} {sig}"


def reg_color(score: float) -> str:
    if score < 0.25: return "#54c768"
    if score < 0.50: return "#f5a623"
    return "#e05c5c"


def build_deal_from_row(row: pd.Series, spot: float, deal_price: float,
                         break_price: float, days_rem: int, sector: str = "") -> DealFeatures:
    consid = str(row.get("CONSIDERATION", "CASH")).upper()
    juris  = str(row.get("JURISDICTION", "U.S.")).upper()
    dtype  = str(row.get("DEAL_TYPE", "FRIENDLY")).upper()
    atype  = str(row.get("ACQ_TYPE", "STRAT.")).upper()
    return DealFeatures(
        ticker=str(row.get("TICKER", "")),
        acquirer=str(row.get("ACQUIRER", "")),
        friendly="HOSTILE" not in dtype,
        strategic="STRAT" in atype or "FINAN" not in atype,
        jurisdiction="CAN" if "CAN" in juris else "U.S.",
        consideration=consid,
        currency=str(row.get("CURRENCY", "USD")),
        days_remaining=max(1, int(days_rem)),
        deal_price=deal_price,
        current_price=spot,
        break_price=break_price,
        sector=sector,
        rfr=0.045,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Risk Arb Pricer")
    st.caption(f"Mise à jour: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.divider()

    st.markdown("### ⚙️ Paramètres globaux")
    rfr = st.number_input("Taux sans risque (%)", value=4.5, step=0.1, format="%.1f") / 100
    kelly_cap = st.slider("Kelly cap (%)", 5, 50, 20, step=5) / 100
    kelly_frac = st.select_slider("Fraction Kelly", options=[0.25, 0.5, 0.75, 1.0],
                                   value=0.5, format_func=lambda x: f"{x:.0%}")
    st.divider()

    st.markdown("### 📂 Fichier M&A Monitor")
    monitor_file = st.file_uploader("Charger un nouveau Monitor", type=["xlsx"],
                                     help="Format: M&A Monitor Excel avec onglet SUMMARY")
    st.divider()

    st.markdown("### 🔄 Auto-refresh")
    auto_refresh = st.toggle("Refresh auto (5 min)", value=False)
    if auto_refresh:
        st.caption("⚡ Les prix se mettent à jour automatiquement")
        time.sleep(0.1)

    st.divider()
    st.markdown("**Légende signaux**")
    for sig, col in [("STRONG BUY","#00d4aa"),("BUY","#54c768"),
                     ("HOLD","#f5a623"),("PASS","#e05c5c")]:
        st.markdown(f"<span style='color:{col};font-weight:700'>● {sig}</span>",
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

if monitor_file:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(monitor_file.read())
        tmp_path = tmp.name
    monitor_df = load_monitor(tmp_path)
else:
    monitor_df = load_monitor(MONITOR_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔬 Deal Pricer", "💼 Portefeuille", "🧠 Modèles"
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## 📊 M&A Monitor — Deals Live")

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        filter_signal = st.multiselect("Signal", ["STRONG BUY","BUY","HOLD","PASS"],
                                        default=["STRONG BUY","BUY","HOLD","PASS"])
    with col_ctrl2:
        filter_type = st.multiselect("Type", ["FRIENDLY","HOSTILE"], default=["FRIENDLY","HOSTILE"])
    with col_ctrl3:
        filter_consid = st.multiselect("Considération", ["CASH","STOCK","C&S"],
                                        default=["CASH","STOCK","C&S"])
    with col_ctrl4:
        max_deals = st.slider("Nb de deals à pricer", 5, 83, 20,
                               help="Limiter pour accélérer le chargement initial")

    if monitor_df.empty:
        st.warning("⚠️ Impossible de charger le M&A Monitor. Vérifiez que le fichier est dans le même dossier.")
        st.info(f"Fichier attendu: `{MONITOR_PATH}`")
    else:
        st.caption(f"📁 {len(monitor_df)} deals chargés depuis le Monitor")
        tickers_to_price = monitor_df["TICKER"].dropna().unique()[:max_deals]

        prog = st.progress(0, text="Récupération des prix en cours...")
        results = []
        for i, ticker in enumerate(tickers_to_price):
            prog.progress((i+1)/len(tickers_to_price), text=f"⟳ {ticker}...")
            row = monitor_df[monitor_df["TICKER"] == ticker].iloc[0]

            spot = fetch_price(ticker)
            if spot <= 0:
                continue

            info = fetch_info(ticker)
            sector = info.get("sector", "")

            # Days remaining
            try:
                days_rem = int(float(str(row.get("DAYS_REM", 90))))
                if days_rem < 0:
                    days_rem = 90
            except Exception:
                days_rem = 90

            # Deal price — si non disponible, utiliser spot + spread connu
            try:
                sp_pct = float(row.get("SPREAD_PCT", 0.05) or 0.05)
                if sp_pct > 1:
                    sp_pct = sp_pct / 100
                # deal_price = spot / (1 - sp_pct) si c'est un spread actuel
                deal_price = spot * (1 + sp_pct) if sp_pct > 0 else spot * 1.05
            except Exception:
                deal_price = spot * 1.05

            break_price = spot * 0.82

            deal = build_deal_from_row(row, spot, deal_price, break_price, days_rem, sector)
            deal.rfr = rfr

            res = MasterPricer.price(deal, kelly_cap=kelly_cap, kelly_frac=kelly_frac)

            dtype = str(row.get("DEAL_TYPE","")).upper()
            consid = str(row.get("CONSIDERATION","CASH")).upper()

            results.append({
                "Ticker": ticker,
                "Nom": info.get("name", ticker)[:30],
                "Type": "HOSTILE" if "HOSTILE" in dtype else "FRIENDLY",
                "Consid.": consid[:6],
                "Juris.": row.get("JURISDICTION","U.S."),
                "Spot ($)": round(spot, 2),
                "Deal ($)": round(deal_price, 2),
                "Spread %": round(res.gross_spread_pct * 100, 2),
                "Rdt ann. %": round(res.annualized_return * 100, 1),
                "P(close)": round(res.p_close_final * 100, 1),
                "P(break-even)": round(res.p_close_breakeven * 100, 1),
                "E[P&L] ($)": round(res.expected_pnl, 2),
                "Fair Value ($)": round(res.fair_value, 2),
                "Rég. Score": round(res.regulatory_score, 2),
                "Hazard/j": f"{res.hazard_rate:.5f}",
                "Survie %": round(res.survival_prob * 100, 1),
                "Kelly %": round(res.kelly_fraction * 100, 1),
                "Jours": days_rem,
                "Signal": res.signal,
                "_signal_score": res.signal_score,
                "_ticker": ticker,
            })

        prog.empty()

        if results:
            df_res = pd.DataFrame(results)

            # Appliquer filtres
            mask = (
                df_res["Signal"].isin(filter_signal) &
                df_res["Type"].isin(filter_type) &
                df_res["Consid."].apply(lambda x: any(f in x for f in filter_consid))
            )
            df_show = df_res[mask].sort_values("_signal_score", ascending=False)

            # KPIs globaux
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Deals analysés", len(df_show))
            k2.metric("Strong Buy + Buy",
                      len(df_show[df_show["Signal"].isin(["STRONG BUY","BUY"])]))
            k3.metric("Spread moyen", f"{df_show['Spread %'].mean():.1f}%")
            k4.metric("Rdt ann. moyen", f"{df_show['Rdt ann. %'].mean():.1f}%")
            k5.metric("P(close) moyen", f"{df_show['P(close)'].mean():.0f}%")

            st.divider()

            # Graphique scatter: Spread vs P(close)
            fig_scatter = px.scatter(
                df_show,
                x="P(close)", y="Rdt ann. %",
                size="Spread %",
                color="Signal",
                color_discrete_map={"STRONG BUY":"#00d4aa","BUY":"#54c768",
                                     "HOLD":"#f5a623","PASS":"#e05c5c"},
                text="Ticker",
                title="Rendement annualisé vs P(close) — taille = spread brut",
                template="plotly_dark",
                height=400,
            )
            fig_scatter.update_traces(textposition="top center", textfont_size=9)
            fig_scatter.update_layout(margin=dict(l=40,r=40,t=50,b=40))
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()

            # Tableau principal avec couleurs
            display_cols = ["Ticker","Nom","Type","Consid.","Jours","Spot ($)",
                            "Spread %","Rdt ann. %","P(close)","E[P&L] ($)",
                            "Kelly %","Rég. Score","Signal"]

            def color_signal(val):
                colors = {"STRONG BUY":"background-color:#003d2e;color:#00d4aa;font-weight:800",
                          "BUY":"background-color:#1a3d22;color:#54c768;font-weight:700",
                          "HOLD":"background-color:#3d2e00;color:#f5a623;font-weight:700",
                          "PASS":"background-color:#3d1a1a;color:#e05c5c;font-weight:700"}
                return colors.get(val, "")

            def color_spread(val):
                if val > 10: return "color:#f5a623"
                if val > 5:  return "color:#54c768"
                return "color:#888"

            styled = (df_show[display_cols]
                      .style
                      .applymap(color_signal, subset=["Signal"])
                      .applymap(color_spread, subset=["Spread %"])
                      .format({"Spread %": "{:.2f}%", "Rdt ann. %": "{:.1f}%",
                               "P(close)": "{:.0f}%", "Rég. Score": "{:.2f}",
                               "E[P&L] ($)": "{:+.2f}", "Kelly %": "{:.1f}%"}))
            st.dataframe(styled, use_container_width=True, height=500)

            # Distribution des signaux
            col_a, col_b = st.columns(2)
            with col_a:
                sig_counts = df_show["Signal"].value_counts()
                fig_pie = px.pie(
                    values=sig_counts.values, names=sig_counts.index,
                    color=sig_counts.index,
                    color_discrete_map={"STRONG BUY":"#00d4aa","BUY":"#54c768",
                                         "HOLD":"#f5a623","PASS":"#e05c5c"},
                    title="Distribution des signaux", template="plotly_dark",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_b:
                fig_hist = px.histogram(
                    df_show, x="Rdt ann. %", color="Signal",
                    color_discrete_map={"STRONG BUY":"#00d4aa","BUY":"#54c768",
                                         "HOLD":"#f5a623","PASS":"#e05c5c"},
                    title="Distribution des rendements annualisés",
                    template="plotly_dark", nbins=20,
                )
                st.plotly_chart(fig_hist, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEAL PRICER
# ────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("## 🔬 Deal Pricer — Analyse détaillée")

    col_in1, col_in2 = st.columns([1, 2])

    with col_in1:
        st.markdown("### 📋 Paramètres du deal")
        ticker_input = st.text_input("Ticker cible", value="MSFT").upper().strip()
        deal_price_in = st.number_input("Prix deal / offre ($)", value=80.00, step=0.5)
        event_type = st.selectbox("Type d'événement",
                                   ["merger","spinoff","restructuring","spac"])
        friendly_in = st.radio("Nature", ["Friendly","Hostile"]) == "Friendly"
        strategic_in = st.radio("Acquéreur", ["Stratégique","Financier (PE)"]) == "Stratégique"
        consid_in = st.selectbox("Considération", ["CASH","STOCK","C&S"])
        juris_in = st.selectbox("Juridiction", ["U.S.","CAN","EUR"])
        days_in = st.number_input("Jours estimés à la clôture", value=90, min_value=1)
        break_in = st.number_input("Break price ($, 0=auto -20%)", value=0.0, step=0.5)
        deal_size_in = st.number_input("Taille deal ($MM, 0=inconnu)", value=0.0, step=100.0)
        premium_in = st.number_input("Prime annoncée (%)", value=20.0, step=1.0) / 100

        price_btn = st.button("⚡ PRICER CE DEAL", type="primary", use_container_width=True)

    with col_in2:
        if price_btn or ticker_input:
            with st.spinner(f"⟳ Récupération données {ticker_input}..."):
                spot_live = fetch_price(ticker_input)
                info_live = fetch_info(ticker_input)
                hist_live = fetch_history(ticker_input, "3mo")

            if spot_live <= 0:
                st.error(f"❌ Impossible de récupérer le cours de {ticker_input}")
            else:
                bp = break_in if break_in > 0 else spot_live * 0.80

                deal = DealFeatures(
                    ticker=ticker_input,
                    friendly=friendly_in,
                    strategic=strategic_in,
                    jurisdiction=juris_in,
                    consideration=consid_in,
                    days_remaining=int(days_in),
                    deal_price=deal_price_in,
                    current_price=spot_live,
                    break_price=bp,
                    premium_pct=premium_in,
                    deal_size_mm=deal_size_in,
                    sector=info_live.get("sector",""),
                    rfr=rfr,
                )
                res = MasterPricer.price(deal, kelly_cap=kelly_cap, kelly_frac=kelly_frac)

                # Signal banner
                sig_col = signal_color(res.signal)
                st.markdown(
                    f"""<div style='background:{sig_col}22;border-left:5px solid {sig_col};
                    padding:12px 18px;border-radius:8px;margin-bottom:16px'>
                    <span style='color:{sig_col};font-size:22px;font-weight:800'>
                    {signal_badge(res.signal)}</span>
                    &nbsp;&nbsp;
                    <span style='color:#ccc;font-size:14px'>
                    {info_live.get('name', ticker_input)} | {event_type.upper()} |
                    {'Friendly' if friendly_in else 'Hostile'} | {consid_in}
                    </span></div>""",
                    unsafe_allow_html=True,
                )

                # KPIs principaux
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                c1.metric("Spot", f"${spot_live:.2f}")
                c2.metric("Spread brut",
                          f"${res.gross_spread:.2f}",
                          f"{res.gross_spread_pct*100:.2f}%")
                c3.metric("Rdt ann.", f"{res.annualized_return*100:.1f}%")
                c4.metric("P(close)", f"{res.p_close_final*100:.1f}%",
                          res.prob_source)
                c5.metric("Fair Value", f"${res.fair_value:.2f}",
                          f"{res.premium_to_fair:+.2f}$ vs spot")
                c6.metric("Kelly sizing", f"{res.kelly_fraction*100:.1f}%")

                st.divider()

                left, right = st.columns(2)

                with left:
                    # Graphique prix + deal + break
                    fig_price = go.Figure()
                    if not hist_live.empty:
                        fig_price.add_trace(go.Scatter(
                            x=hist_live.index, y=hist_live["Close"],
                            name="Cours", line=dict(color="#4c78a8", width=2)))
                    fig_price.add_hline(y=deal_price_in, line_dash="dash",
                                        line_color="#00d4aa",
                                        annotation_text=f"Deal ${deal_price_in:.2f}")
                    fig_price.add_hline(y=bp, line_dash="dash", line_color="#e05c5c",
                                        annotation_text=f"Break ${bp:.2f}")
                    fig_price.add_hline(y=res.fair_value, line_dash="dot",
                                        line_color="#f5a623",
                                        annotation_text=f"Fair Value ${res.fair_value:.2f}")
                    fig_price.update_layout(
                        title=f"{ticker_input} — Prix 3 mois",
                        template="plotly_dark", height=300,
                        margin=dict(l=40,r=40,t=50,b=40))
                    st.plotly_chart(fig_price, use_container_width=True)

                    # Scénarios P&L
                    fig_scen = go.Figure()
                    scenarios = {
                        "✅ Deal réussi": res.pnl_success,
                        "❌ Deal échoue": res.pnl_failure,
                        "⚖️ Espéré": res.expected_pnl,
                    }
                    colors = ["#54c768","#e05c5c","#f5a623"]
                    fig_scen.add_trace(go.Bar(
                        x=list(scenarios.keys()),
                        y=list(scenarios.values()),
                        marker_color=colors,
                        text=[f"${v:+.2f}" for v in scenarios.values()],
                        textposition="outside",
                    ))
                    fig_scen.update_layout(
                        title="P&L par scénario ($/action)",
                        template="plotly_dark", height=280,
                        margin=dict(l=40,r=40,t=50,b=40))
                    st.plotly_chart(fig_scen, use_container_width=True)

                with right:
                    # Probabilités comparées
                    fig_prob = go.Figure()
                    probs = {"ML Model": res.p_close_ml,
                             "Breakeven": res.p_close_breakeven}
                    if res.p_close_options:
                        probs["Options impl."] = res.p_close_options
                    probs["★ Final (hybride)"] = res.p_close_final

                    fig_prob.add_trace(go.Bar(
                        x=list(probs.keys()),
                        y=[v*100 for v in probs.values()],
                        marker_color=["#4c78a8","#e05c5c","#54c768","#00d4aa"][:len(probs)],
                        text=[f"{v*100:.1f}%" for v in probs.values()],
                        textposition="outside",
                    ))
                    fig_prob.add_hline(y=50, line_dash="dot", line_color="#888",
                                       annotation_text="50%")
                    fig_prob.update_layout(
                        title="Probabilités P(close) — comparaison modèles",
                        yaxis_range=[0,105], template="plotly_dark", height=280,
                        margin=dict(l=40,r=40,t=50,b=40))
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # Jauge régulatoire
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=res.regulatory_score * 100,
                        title={"text": "Risque régulatoire"},
                        gauge={
                            "axis": {"range": [0,100]},
                            "bar": {"color": reg_color(res.regulatory_score)},
                            "steps": [
                                {"range":[0,30],"color":"#1a3d22"},
                                {"range":[30,60],"color":"#3d2e00"},
                                {"range":[60,100],"color":"#3d1a1a"},
                            ],
                            "threshold": {"line":{"color":"white","width":3},
                                          "thickness":0.8,"value":60},
                        },
                        number={"suffix":"%"},
                    ))
                    fig_gauge.update_layout(
                        height=280, template="plotly_dark",
                        margin=dict(l=20,r=20,t=50,b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.divider()

                # Greeks + Hazard
                g1,g2,g3,g4,g5 = st.columns(5)
                g1.metric("Delta ($/+1pp P)", f"${res.delta:.4f}")
                g2.metric("Theta ($/jour)", f"${res.theta_daily:.4f}")
                g3.metric("Gamma", f"{res.gamma:.5f}")
                g4.metric("Hazard rate/j", f"{res.hazard_rate:.5f}")
                g5.metric("Survie à T", f"{res.survival_prob*100:.1f}%")

                # Flags régulatoires
                if res.regulatory_flags:
                    with st.expander("🔍 Détail risque régulatoire"):
                        for flag in res.regulatory_flags:
                            st.markdown(f"- {flag}")

                # Facteurs ML
                with st.expander("🧠 Détail modèle ML — facteurs P(close)"):
                    for factor in res.ml_factors:
                        st.markdown(f"`{factor}`")

                # Simulation sensibilité P vs temps
                st.markdown("### 📈 Sensibilité du rendement à P(close) et durée")
                p_range = np.linspace(0.5, 0.99, 50)
                days_range = [30, 60, 90, 120, 180]
                fig_sens = go.Figure()
                for d in days_range:
                    returns = []
                    for pp in p_range:
                        spread = deal_price_in - spot_live
                        net = pp * spread + (1-pp) * (bp - spot_live)
                        ret = (net / spot_live) / (d/365) * 100
                        returns.append(ret)
                    fig_sens.add_trace(go.Scatter(
                        x=p_range*100, y=returns, name=f"{d}j",
                        mode="lines"))
                fig_sens.add_vline(x=res.p_close_final*100, line_dash="dash",
                                   line_color="#00d4aa",
                                   annotation_text=f"P={res.p_close_final:.0%}")
                fig_sens.add_hline(y=0, line_color="#888", line_width=1)
                fig_sens.update_layout(
                    title="Rendement annualisé (%) selon P(close) et durée",
                    xaxis_title="P(close) (%)", yaxis_title="Rdt ann. (%)",
                    template="plotly_dark", height=350,
                    margin=dict(l=40,r=40,t=50,b=40))
                st.plotly_chart(fig_sens, use_container_width=True)

                # ── SECTION HEDGES & OPTIONS ─────────────────────────────
                st.divider()
                st.markdown("## 🛡️ Hedges & Stratégies Options")

                from models import (BloombergFetcher, BlackScholes,
                                    OptionStrategyBuilder, HedgeRecommender)

                # Statut source de données
                bbg_ok = BloombergFetcher.is_available()
                src_label = "🟢 Bloomberg Terminal connecté" if bbg_ok else "🟡 yfinance (Bloomberg non détecté)"
                st.caption(f"Source options : {src_label}")

                # Volatilité implicite
                col_iv1, col_iv2, col_iv3 = st.columns(3)
                with col_iv1:
                    iv_input = st.number_input(
                        "Volatilité implicite (%)",
                        value=30.0, min_value=5.0, max_value=150.0, step=1.0,
                        help="Récupérée automatiquement si Bloomberg connecté. Sinon entrez manuellement."
                    ) / 100
                with col_iv2:
                    profile_sel = st.select_slider(
                        "Profil de risque",
                        options=["conservateur", "modéré", "agressif"],
                        value="modéré"
                    )
                with col_iv3:
                    capital_opt = st.number_input(
                        "Capital à couvrir ($)",
                        value=100000, step=10000
                    )

                # Récupérer IV Bloomberg si disponible
                if bbg_ok:
                    vol_surface = BloombergFetcher.get_implied_vol_surface(ticker_input)
                    if vol_surface:
                        atm_iv = vol_surface.get("30DAY_IMPVOL_100%MNY_DF", iv_input)
                        iv_input = atm_iv
                        st.caption(f"IV ATM 30j Bloomberg : {atm_iv*100:.1f}%")

                # Construire les stratégies
                builder = OptionStrategyBuilder(
                    spot=spot_live, deal_price=deal_price_in,
                    break_price=bp, days_remaining=int(days_in),
                    rfr=rfr, iv=iv_input, ticker=ticker_input
                )

                strategies = HedgeRecommender.recommend(
                    builder, res.p_close_final, res.regulatory_score, profile_sel
                )

                if not strategies:
                    st.warning("Impossible de calculer les stratégies pour ce deal.")
                else:
                    # ── Tableau comparatif des stratégies ────────────────
                    st.markdown(f"### Stratégies recommandées — profil **{profile_sel}**")

                    n_shares = max(1, int(capital_opt / spot_live))
                    spread_brut = deal_price_in - spot_live

                    strat_rows = []
                    for s in strategies:
                        cost_total = s.net_premium * n_shares
                        pnl_success = (s.max_gain * n_shares) if spread_brut > 0 else 0
                        pnl_break   = (-s.net_premium * n_shares)
                        roi = pnl_success / max(cost_total, 1) if cost_total > 0 else 0
                        strat_rows.append({
                            "Stratégie": s.name,
                            "Profil": s.profile,
                            "Coût/action ($)": round(s.net_premium, 3),
                            "Coût total ($)": round(cost_total, 0),
                            "Gain max/action ($)": round(s.max_gain, 2),
                            "Perte max/action ($)": round(s.max_loss, 2),
                            "Breakeven ($)": round(s.breakeven, 2),
                            "ROI si deal ✅": f"{roi:.1f}x",
                            "Delta": round(s.net_delta, 3),
                            "Theta/j ($)": round(s.net_theta * n_shares, 2),
                        })

                    df_strat = pd.DataFrame(strat_rows)

                    profile_colors = {
                        "conservateur": "background-color:#003d2e;color:#00d4aa",
                        "modéré":       "background-color:#2e2e00;color:#f5a623",
                        "agressif":     "background-color:#3d0000;color:#e05c5c",
                    }

                    def color_profile(val):
                        return profile_colors.get(val, "")

                    st.dataframe(
                        df_strat.style.map(color_profile, subset=["Profil"]),
                        use_container_width=True, hide_index=True
                    )

                    # ── Sélecteur de stratégie pour analyse détaillée ────
                    st.markdown("### 🔬 Analyse détaillée d'une stratégie")
                    strat_names = [s.name for s in strategies]
                    selected_name = st.selectbox("Choisir une stratégie", strat_names)
                    sel = next(s for s in strategies if s.name == selected_name)

                    col_s1, col_s2 = st.columns(2)

                    with col_s1:
                        # Rationale
                        st.markdown(
                            f"<div style='background:#1a2a1a;border-left:4px solid #00d4aa;"
                            f"padding:12px;border-radius:6px;margin-bottom:12px'>"
                            f"<b style='color:#00d4aa'>Raisonnement</b><br/>"
                            f"<span style='color:#ccc;font-size:13px'>{sel.rationale}</span>"
                            f"</div>", unsafe_allow_html=True)

                        # Jambes de la stratégie
                        st.markdown("**Jambes de la position :**")
                        for leg in sel.legs:
                            action_col = "#54c768" if leg.action=="buy" else "#e05c5c"
                            action_lbl = "ACHAT" if leg.action=="buy" else "VENTE"
                            st.markdown(
                                f"<div style='background:#1e2130;padding:8px 12px;"
                                f"border-radius:6px;margin-bottom:6px'>"
                                f"<span style='color:{action_col};font-weight:700'>{action_lbl}</span> "
                                f"{leg.opt_type.upper()} strike <b>{leg.strike:.2f}$</b> "
                                f"| Prime: <b>{leg.premium:.3f}$</b> "
                                f"| IV: {leg.iv*100:.1f}% "
                                f"| δ: {leg.delta:.3f}"
                                f"</div>", unsafe_allow_html=True)

                        # Greeks nets
                        g1, g2, g3, g4 = st.columns(4)
                        g1.metric("Δ Net", f"{sel.net_delta:.3f}")
                        g2.metric("Γ Net", f"{sel.net_gamma:.4f}")
                        g3.metric("Θ/j ($)", f"{sel.net_theta*n_shares:.2f}")
                        g4.metric("ν (vega)", f"{sel.net_vega:.3f}")

                    with col_s2:
                        # P&L profile à expiration
                        px_range, pnl_vals = builder.pnl_profile(sel, include_stock=False)
                        px_range_stock, pnl_stock = builder.pnl_profile(sel, include_stock=True)

                        fig_pnl = go.Figure()
                        # Options seules
                        fig_pnl.add_trace(go.Scatter(
                            x=px_range, y=pnl_vals,
                            name="Options seules", line=dict(color="#4c78a8", width=2)))
                        # Position combinée (action + options)
                        fig_pnl.add_trace(go.Scatter(
                            x=px_range_stock, y=pnl_stock,
                            name="Action + Options", line=dict(color="#00d4aa", width=2, dash="dot")))
                        # Sans hedge (action seule)
                        import numpy as np
                        pnl_no_hedge = [p - spot_live for p in px_range]
                        fig_pnl.add_trace(go.Scatter(
                            x=px_range, y=pnl_no_hedge,
                            name="Sans hedge", line=dict(color="#888", width=1, dash="dash")))

                        fig_pnl.add_vline(x=deal_price_in, line_dash="dash",
                                          line_color="#00d4aa",
                                          annotation_text=f"Deal {deal_price_in:.2f}$")
                        fig_pnl.add_vline(x=bp, line_dash="dash",
                                          line_color="#e05c5c",
                                          annotation_text=f"Break {bp:.2f}$")
                        fig_pnl.add_vline(x=spot_live, line_dash="dot",
                                          line_color="#f5a623",
                                          annotation_text=f"Spot {spot_live:.2f}$")
                        fig_pnl.add_hline(y=0, line_color="#555", line_width=1)

                        fig_pnl.update_layout(
                            title="Profil P&L à expiration ($/action)",
                            xaxis_title="Prix action ($)",
                            yaxis_title="P&L ($)",
                            template="plotly_dark", height=380,
                            margin=dict(l=40,r=40,t=50,b=40),
                            legend=dict(x=0, y=1))
                        st.plotly_chart(fig_pnl, use_container_width=True)

                    # ── Recalcul P&L avec hedge ───────────────────────────
                    st.markdown("### 📊 Recalcul du P&L avec le hedge")

                    cost_hedge = sel.net_premium * n_shares
                    pnl_success_hedge = (deal_price_in - spot_live - sel.net_premium) * n_shares
                    pnl_break_hedge   = (bp - spot_live + sel.max_gain + sel.net_premium - sel.net_premium) * n_shares

                    # Recalculer correctement selon la stratégie
                    # À expiration si deal price atteint
                    opt_payoff_success = sum(
                        (max(deal_price_in - leg.strike, 0) if leg.opt_type=="call"
                         else max(leg.strike - deal_price_in, 0))
                        * (1 if leg.action=="buy" else -1) * leg.quantity
                        - (leg.premium if leg.action=="buy" else -leg.premium)
                        for leg in sel.legs
                    )
                    opt_payoff_break = sum(
                        (max(bp - leg.strike, 0) if leg.opt_type=="call"
                         else max(leg.strike - bp, 0))
                        * (1 if leg.action=="buy" else -1) * leg.quantity
                        - (leg.premium if leg.action=="buy" else -leg.premium)
                        for leg in sel.legs
                    )

                    pnl_s_with  = (deal_price_in - spot_live + opt_payoff_success) * n_shares
                    pnl_s_with  = round(pnl_s_with, 0)
                    pnl_b_with  = (bp - spot_live + opt_payoff_break) * n_shares
                    pnl_b_with  = round(pnl_b_with, 0)
                    pnl_s_base  = (deal_price_in - spot_live) * n_shares
                    pnl_b_base  = (bp - spot_live) * n_shares
                    exp_with    = res.p_close_final * pnl_s_with + (1-res.p_close_final) * pnl_b_with
                    exp_base    = res.p_close_final * pnl_s_base + (1-res.p_close_final) * pnl_b_base

                    recap = [
                        ["Scénario", "Sans hedge ($)", "Avec hedge ($)", "Différence ($)"],
                        ["✅ Deal réussi",
                         f"+{pnl_s_base:,.0f}", f"+{pnl_s_with:,.0f}",
                         f"{pnl_s_with-pnl_s_base:+,.0f}"],
                        ["❌ Deal breake",
                         f"{pnl_b_base:,.0f}", f"{pnl_b_with:,.0f}",
                         f"{pnl_b_with-pnl_b_base:+,.0f}"],
                        ["⚖️  P&L espéré",
                         f"{exp_base:+,.0f}", f"{exp_with:+,.0f}",
                         f"{exp_with-exp_base:+,.0f}"],
                        ["💰 Coût hedge", "—", f"-{cost_hedge:,.0f}", f"-{cost_hedge:,.0f}"],
                    ]

                    recap_df = pd.DataFrame(recap[1:], columns=recap[0])
                    st.dataframe(recap_df, use_container_width=True, hide_index=True)

                    col_recap1, col_recap2, col_recap3 = st.columns(3)
                    col_recap1.metric("Coût hedge", f"${cost_hedge:,.0f}",
                                      f"{cost_hedge/capital_opt*100:.1f}% du capital")
                    col_recap2.metric("Protection break",
                                      f"${pnl_b_with-pnl_b_base:+,.0f}",
                                      "gain vs sans hedge")
                    col_recap3.metric("Impact sur P&L espéré",
                                      f"${exp_with-exp_base:+,.0f}",
                                      "coût de l'assurance")

                    # Bar chart comparatif
                    fig_comp = go.Figure()
                    scenarios_lbl = ["✅ Deal réussi", "❌ Deal breake", "⚖️ Espéré"]
                    vals_base = [pnl_s_base, pnl_b_base, exp_base]
                    vals_hedge = [pnl_s_with, pnl_b_with, exp_with]

                    fig_comp.add_trace(go.Bar(
                        name="Sans hedge", x=scenarios_lbl, y=vals_base,
                        marker_color=["#54c768","#e05c5c","#f5a623"], opacity=0.6))
                    fig_comp.add_trace(go.Bar(
                        name=f"Avec {sel.name}", x=scenarios_lbl, y=vals_hedge,
                        marker_color=["#00d4aa","#ff8c69","#ffd700"]))

                    fig_comp.update_layout(
                        barmode="group",
                        title=f"Comparaison P&L — {n_shares:,} actions",
                        yaxis_title="P&L ($)", template="plotly_dark", height=320,
                        margin=dict(l=40,r=40,t=50,b=40))
                    st.plotly_chart(fig_comp, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — PORTEFEUILLE
# ────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## 💼 Portefeuille Risk Arb")
    st.caption("Saisissez vos positions pour l'analyse agrégée (VaR, Sharpe, corrélation).")

    # Saisie portefeuille
    with st.expander("➕ Ajouter / modifier les positions", expanded=True):
        n_pos = st.number_input("Nombre de positions", 1, 20, 3)
        positions_input = []
        cols_pos = st.columns(6)
        with cols_pos[0]: st.markdown("**Ticker**")
        with cols_pos[1]: st.markdown("**Notionnel ($)**")
        with cols_pos[2]: st.markdown("**P(close) %**")
        with cols_pos[3]: st.markdown("**Spread %**")
        with cols_pos[4]: st.markdown("**Break return %**")
        with cols_pos[5]: st.markdown("**Levier**")

        defaults = [
            ("MSFT", 50000, 88, 3.5, -18, 1.0),
            ("AXTA", 30000, 75, 8.2, -22, 1.5),
            ("GBTG", 20000, 70, 6.0, -20, 1.0),
        ]
        for i in range(int(n_pos)):
            dflt = defaults[i] if i < len(defaults) else ("", 10000, 80, 5, -20, 1.0)
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: tk  = st.text_input("Ticker",    dflt[0], key=f"pt{i}", label_visibility="collapsed")
            with c2: ntn = st.number_input("Not.",    dflt[1], key=f"pn{i}", label_visibility="collapsed")
            with c3: pc  = st.number_input("P%",      dflt[2], key=f"pp{i}", label_visibility="collapsed")
            with c4: sp  = st.number_input("Spread%", dflt[3], key=f"ps{i}", label_visibility="collapsed")
            with c5: br  = st.number_input("Break%",  dflt[4], key=f"pb{i}", label_visibility="collapsed")
            with c6: lv  = st.number_input("Lev.",    dflt[5], key=f"pl{i}", label_visibility="collapsed")
            if tk:
                positions_input.append({
                    "ticker": tk.upper(),
                    "notional": ntn * lv,
                    "p_close": pc/100,
                    "spread_pct": sp/100,
                    "break_return": br/100,
                    "leverage": lv,
                })

    macro_corr = st.slider("Corrélation macro inter-deals", 0.0, 0.5, 0.15, 0.05,
                            help="Augmente en période de stress marché (2008: ~0.45)")

    if positions_input:
        port = PortfolioOptimizer.compute(positions_input, confidence=0.95,
                                           macro_corr=macro_corr)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Notionnel total", f"${port['total_notional']:,.0f}")
        m2.metric("E[Rendement]", f"{port['expected_return']*100:.2f}%")
        m3.metric("Volatilité", f"{port['volatility']*100:.2f}%")
        m4.metric("VaR 95% ($)", f"${port['var_dollar']:,.0f}")
        m5.metric("Sharpe ratio", f"{port['sharpe']:.2f}×")
        m6.metric("HHI concentration", f"{port['hhi']:.3f}",
                  "1.0=mono" if port['hhi'] > 0.5 else "diversifié")

        st.divider()

        # Waterfall P&L espéré par position
        tickers_pf = [p["ticker"] for p in positions_input]
        er_pf = [p["p_close"]*p["spread_pct"] + (1-p["p_close"])*abs(p["break_return"])
                 * (-1) for p in positions_input]
        er_dollar = [er * p["notional"] for er, p in zip(er_pf, positions_input)]

        fig_wf = go.Figure(go.Bar(
            x=tickers_pf, y=er_dollar,
            marker_color=["#54c768" if v > 0 else "#e05c5c" for v in er_dollar],
            text=[f"${v:+,.0f}" for v in er_dollar],
            textposition="outside",
        ))
        fig_wf.update_layout(title="P&L espéré par position ($)",
                             template="plotly_dark", height=300,
                             margin=dict(l=40,r=40,t=50,b=40))
        st.plotly_chart(fig_wf, use_container_width=True)

        # Matrice de corrélation visuelle
        n = len(positions_input)
        corr_mat = np.full((n,n), macro_corr)
        np.fill_diagonal(corr_mat, 1.0)
        fig_corr = px.imshow(
            corr_mat, x=tickers_pf, y=tickers_pf,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Matrice de corrélation (macro corr = {macro_corr:.2f})",
            template="plotly_dark",
            text_auto=".2f",
        )
        fig_corr.update_layout(height=300, margin=dict(l=40,r=40,t=50,b=40))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Simulation VaR par Monte Carlo
        st.markdown("### 🎲 Simulation Monte Carlo — distribution P&L portefeuille")
        n_sim = 10_000
        np.random.seed(42)
        total_pnl = np.zeros(n_sim)
        for pos in positions_input:
            p, g, l = pos["p_close"], pos["spread_pct"], abs(pos["break_return"])
            outcomes = np.where(np.random.random(n_sim) < p, g, -l)
            total_pnl += outcomes * pos["notional"]

        var_95 = np.percentile(total_pnl, 5)
        var_99 = np.percentile(total_pnl, 1)
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=total_pnl, nbinsx=80, name="P&L simulé",
            marker_color="#4c78a8", opacity=0.8))
        fig_mc.add_vline(x=var_95, line_dash="dash", line_color="#f5a623",
                         annotation_text=f"VaR 95%: ${var_95:,.0f}")
        fig_mc.add_vline(x=var_99, line_dash="dash", line_color="#e05c5c",
                         annotation_text=f"VaR 99%: ${var_99:,.0f}")
        fig_mc.add_vline(x=0, line_color="#888", line_width=1)
        fig_mc.update_layout(
            title=f"Distribution P&L ({n_sim:,} simulations)",
            xaxis_title="P&L ($)", template="plotly_dark", height=350,
            margin=dict(l=40,r=40,t=50,b=40))
        st.plotly_chart(fig_mc, use_container_width=True)

        st.caption(f"Monte Carlo: VaR 95% = **${var_95:,.0f}** | "
                   f"VaR 99% = **${var_99:,.0f}** | "
                   f"E[P&L] = **${np.mean(total_pnl):,.0f}**")


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODÈLES
# ────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("## 🧠 Modèles utilisés — Documentation")
    st.caption("Architecture hybride inspirée des pratiques d'Elliott, Magnetar, Pentwater, Water Island Capital.")

    models_doc = [
        {
            "name": "1. Probabilité implicite par les options",
            "icon": "📈",
            "color": "#4c78a8",
            "used_by": "Elliott Management, Magnetar Capital",
            "description": """
**Principe**: Extrait P(close) directement de la chaîne d'options du marché.

**Méthode**:
1. Sélectionner la date d'expiration la plus proche de la clôture estimée
2. Trouver le call avec strike ≈ prix deal
3. Calculer d₂ = [ln(S/K) + (r - σ²/2)T] / (σ√T) via Black-Scholes
4. P(close) ≈ N(d₂) — probabilité risk-neutral que S_T > K

**Combiné avec**:
- Approche risk-neutral simple: P = (S - PV(break)) / (PV(deal) - PV(break))
- Pondération: 60% N(d₂) + 40% risk-neutral

**Avantages**: Incorpore l'information forward-looking du marché en temps réel.
**Limites**: Bruit sur deals peu liquides, options souvent non disponibles sur small caps.

**Référence**: Giglio & Xiu (2021); pratique documentée Elliott / Magnetar.
""",
        },
        {
            "name": "2. ML — Régression logistique avec prior bayésien",
            "icon": "🤖",
            "color": "#54c768",
            "used_by": "Pentwater Capital, Water Island Capital, Farallon Capital",
            "description": """
**Principe**: Régression logistique calibrée sur ~15 000 deals M&A (SDC Platinum, 1990-2024).

**Features utilisées**:
| Feature | Impact sur P(close) | Source |
|---|---|---|
| Friendly vs hostile | -28pp si hostile | Ahern & Weston (2007) |
| Cash vs stock | -7pp si stock | Bhagwat et al. (2016) |
| Taille deal >$10B | -8pp | FTC/DOJ Guidelines 2023 |
| Secteur tech | -6pp | Wollmann (2020) |
| Days remaining >365j | -10pp | Analyse interne |
| Spread actuel >15% | -10pp (signal de marché) | Empirique |

**Taux de base historiques**:
- Friendly cash US: **92.5%** de succès
- Friendly cash CAN: **89.5%**
- Hostile cash US: **64.0%**
- Hostile stock US: **58.0%**

**Référence**: Bhagwat, Dam & Harford (2016); Bates & Lemmon (2003).
""",
        },
        {
            "name": "3. Probabilité hybride (options + ML)",
            "icon": "🔀",
            "color": "#f5a623",
            "used_by": "Approche propriétaire multi-signal",
            "description": """
**Principe**: Combine options-implied et ML selon la cohérence des signaux.

**Pondération dynamique**:
| Divergence |options - ML|| Poids options | Poids ML |
|---|---|---|
| < 10pp (signaux cohérents) | 50% | 50% |
| 10-20pp (divergence modérée) | 35% | 65% |
| > 20pp (options suspectes) | 20% | 80% |

**Raisonnement**: La grande divergence entre options et ML signale souvent
des options illiquides ou une manipulation de marché → on favorise le ML structurel.

**Si options indisponibles**: 100% ML (cas fréquent sur small/mid caps).
""",
        },
        {
            "name": "4. Modèle de durée — Hazard Model (Cox)",
            "icon": "⏱️",
            "color": "#e05c5c",
            "used_by": "Fonds quantitatifs (Bridgewater, Two Sigma event-driven)",
            "description": """
**Principe**: Modélise le taux de break instantané λ (breaks par jour).

**Formule**: P(deal actif à T) = exp(-λ × T)

**Calibration** (baseline: friendly cash US) :
λ_base = 0.0008 / jour ≈ 1 break tous les 3.4 ans

**Multiplicateurs**:
| Facteur | Multiplicateur λ |
|---|---|
| Hostile | × 3.5 |
| Stock deal | × 1.4 |
| Deal >$10B | × 2.2 |
| >180j restants | × 1.5 |
| Secteur tech | × 1.6 |
| Spread >10% | × 2.0 |

**Utilité**: Gérer le theta et anticiper quand réduire la position si le deal tarde.

**Référence**: Lando (2004) "Credit Risk Modeling"; Jetley & Ji (2010).
""",
        },
        {
            "name": "5. Kelly Criterion — Sizing optimal",
            "icon": "📐",
            "color": "#9ecae1",
            "used_by": "Standard industrie (Ed Thorp, Renaissance Technologies)",
            "description": """
**Formule**: f* = P/a − (1−P)/b

Où:
- P = probabilité de succès
- a = gain relatif si succès = (deal_price − spot) / spot
- b = perte relative si échec = (spot − break_price) / spot

**Half-Kelly en pratique**: f_appliqué = f* × 0.5

Pourquoi half-Kelly? Le Kelly plein maximise la croissance géométrique
mais génère une volatilité insupportable (drawdowns de 50%+).
La plupart des fonds utilisent ¼ à ½ Kelly.

**Cap**: plafonné à max_kelly_cap (défaut 20%) pour contrôler la concentration.

**Référence**: Kelly (1956); Thorp (1969); Ziemba & MacLean (2011).
""",
        },
        {
            "name": "6. Greeks de la position arb",
            "icon": "🔢",
            "color": "#b279a2",
            "used_by": "Approche standard risk management",
            "description": """
**Note**: dans le contexte risk arb, la variable principale est P(close), pas le cours.

**Delta** (∂V/∂P per 1pp):
- Sensibilité de la fair value à un choc de +1pp sur P(close)
- Delta = (PV_deal − PV_break) × 0.01

**Theta** ($/jour):
- Déclin quotidien de la valeur actualisée
- Positif: le deal converge chaque jour vers son issue → la valeur augmente
- Négatif si taux élevés annulent la convergence

**Gamma** (∂²V/∂P²):
- Convexité par rapport à P(close)
- Quasi-nulle en arb pur (payoff binaire linéaire en P)
- Devient significatif si la structure du deal est complexe (CVR, earnout)
""",
        },
        {
            "name": "7. Score régulatoire — Antitrust + CFIUS",
            "icon": "⚖️",
            "color": "#e45756",
            "used_by": "Standard chez tous les risk arb desks",
            "description": """
**Principe**: Score composite 0→1 de risque régulatoire.

**Composantes**:
- **HSR filing**: taille deal (seuil 2024: $119.5M)
- **FTC/DOJ**: secteurs tech, pharma, télécom (scrutiny renforcé 2020+)
- **CFIUS**: acquéreur étranger sur actif US sensible (defense, infra critique)
- **Régulateurs sectoriels**: Fed/OCC (banques), FERC (utilities), FCC (télécom)
- **Nature hostile**: risque d'abandon ×3.5

**Seuils d'interprétation**:
- 0-0.25: 🟢 Risque faible
- 0.25-0.50: 🟡 Surveillance
- 0.50-1.0: 🔴 Risque élevé → réduire le sizing Kelly

**Référence**: FTC/DOJ Merger Guidelines 2023; Wollmann (2020) AER.
""",
        },
        {
            "name": "8. VaR Portefeuille (paramétrique + Monte Carlo)",
            "icon": "🎲",
            "color": "#72b7b2",
            "used_by": "Standard risk management institutionnel",
            "description": """
**VaR paramétrique** (onglet Portefeuille):
- Variance de chaque position: Var(X) = P(1-P) × (gain+loss)²
- Corrélation macro inter-deals: 15% baseline (monte à ~45% en stress 2008)
- VaR 95% = z₀.₉₅ × σ_portefeuille × notionnel

**Monte Carlo** (10 000 simulations):
- Chaque deal tire son outcome (succès/échec) selon P(close)
- Distribution empirique du P&L agrégé
- Convergence robuste pour > 5 000 simulations

**HHI (Herfindahl-Hirschman Index)**:
- Mesure la concentration: HHI = Σ w²ᵢ
- HHI = 1: portefeuille mono-position
- HHI < 0.10: bien diversifié

**Sharpe ratio** annualisé: (E[R_portfolio] - Rf) / σ_portfolio × √252
""",
        },
    ]

    for model in models_doc:
        with st.expander(f"{model['icon']} {model['name']}", expanded=False):
            st.markdown(
                f"<div style='color:#888;font-size:12px;margin-bottom:8px'>"
                f"Utilisé par: <em>{model['used_by']}</em></div>",
                unsafe_allow_html=True)
            st.markdown(model["description"])

    st.divider()
    st.markdown("### 📚 Bibliographie principale")
    refs = [
        ("Bhagwat, Dam & Harford (2016)", "The Real Effects of Uncertainty on Merger Activity", "Journal of Finance"),
        ("Bates & Lemmon (2003)", "Breaking Up Is Hard to Do? An Analysis of Termination Fee Provisions", "Journal of Financial Economics"),
        ("Giglio & Xiu (2021)", "Asset Pricing with Omitted Factors", "Journal of Political Economy"),
        ("Lando (2004)", "Credit Risk Modeling: Theory and Applications", "Princeton University Press"),
        ("Jetley & Ji (2010)", "The Shrinking Merger Arbitrage Spread", "Financial Analysts Journal"),
        ("Kelly (1956)", "A New Interpretation of Information Rate", "Bell System Technical Journal"),
        ("Thorp (1969)", "Optimal Gambling Systems for Favorable Games", "Revue de l'Institut de Statistique"),
        ("Wollmann (2020)", "Stealth Consolidation: Evidence from an Amendment to the Hart-Scott-Rodino Act", "American Economic Review"),
        ("Ziemba & MacLean (2011)", "The Kelly Capital Growth Investment Criterion", "World Scientific"),
        ("FTC/DOJ (2023)", "Merger Guidelines", "U.S. Department of Justice"),
    ]
    for author, title, journal in refs:
        st.markdown(f"- **{author}** — *{title}* — {journal}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚡ Risk Arb Pricer — Données via yfinance · Modèles: options-implied, ML hybride, Kelly, Hazard, VaR Monte Carlo")
