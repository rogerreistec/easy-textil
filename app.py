# app.py
# Easy Textil — CSV normalizador com OEE, mapeamento assistido e design aprimorado

import io
import re
import os
import json
import zipfile
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# ARQUIVOS DE ASSETS
# ============================================================
LOGO_PATH = "assets/logo.jpg"
HERO_PATH = "assets/hero.jpg"

if not os.path.exists(LOGO_PATH):
    raise FileNotFoundError("❌ Coloque sua logo em 'assets/logo.jpg'.")

logo_image = Image.open(LOGO_PATH)
hero_image = Image.open(HERO_PATH) if os.path.exists(HERO_PATH) else None

# ============================================================
# CONFIG DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon=logo_image,
    layout="wide",
)

# ============================================================
# CSS ESTILIZADO
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; max-width: 1400px;}
    h1 { font-weight: 800; letter-spacing: -0.5px; }
    h3 { font-weight: 600; }
    .small-muted { color:#6b7280; font-size:0.92rem; }
    .card {
        border:1px solid #e5e7eb; border-radius:12px; padding:16px;
        background:white; box-shadow:0 1px 2px rgba(0,0,0,0.05);
        text-align:center;
    }
    .card h3 { margin:0; font-size:0.95rem; color:#6b7280; font-weight:600; }
    .card .value { font-size:1.4rem; font-weight:800; margin-top:.3rem; color:#111827; }
    .hero {
        border-radius:16px; padding:20px;
        background:linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        text-align:center;
        margin-bottom: 1rem;
    }
    .hero h1 { margin:0; color:#1E3A8A; font-weight:800; }
    .hero .sub { color:#334155; font-size:1rem; }
    .chip { display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .6rem;
            border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem; }
    .chip-ok { background:#DCFCE7; color:#065F46; border:1px solid #86EFAC; }
    .chip-bad { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR (logo + upload + passo a passo)
# ============================================================
st.sidebar.image(logo_image, use_container_width=True)
st.sidebar.caption("Seu medidor de eficiência")

with st.sidebar.expander("📋 Passo a passo (bem simples)", expanded=True):
    st.markdown(
        """
        1️⃣ **Baixe o CSV Modelo**.  
        2️⃣ **Carregue um CSV** (ou **.zip** com vários CSVs).  
        3️⃣ **Ajuste o mapeamento** (aba 🧭).  
        4️⃣ Veja **Dados Limpos** e **OEE**.  
        5️⃣ **Baixe os resultados**.
        """
    )

MODEL_DF = pd.DataFrame({
    "CLIENTE": ["Exemplo Indústria"],
    "QUANTIDADE_PRODUTO": [1000],
    "MAQUINAS_NECESSARIAS": [1],
    "DATA_INICIO": ["2025-01-07"],
    "TEMPO_PARADA_MAQUINA_MIN": [0],
    "QTD_REFUGADA": [0],
})
st.sidebar.download_button(
    "⬇️ Baixar CSV Modelo",
    data=MODEL_DF.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_modelo.csv",
)

uploaded_csv = st.sidebar.file_uploader("📥 CSV único", type=["csv"])
uploaded_zip = st.sidebar.file_uploader("📦 Lote (.zip com vários CSVs)", type=["zip"])

# ============================================================
# ALIASES E FUNÇÕES
# ============================================================
ALIASES = {
    "CLIENTE": ["cliente", "pedido", "cliente/pedido"],
    "QUANTIDADE_PRODUTO": ["quantidade", "kg prod", "kg produzido"],
    "MAQUINAS_NECESSARIAS": ["maquinas necessarias", "maq", "turnos"],
    "DATA_INICIO": ["data inicio", "inicio prod"],
    "TEMPO_PARADA_MAQUINA_MIN": ["parada", "horas maq", "tempo parada"],
    "QTD_REFUGADA": ["refugo", "kg rest"],
    "TURNOS_NECESSARIOS": ["turnos necessarios"],
    "QUANTIDADE_PLANEJADA": ["quantidade planejada", "planejado"],
}
REQUIRED = ["CLIENTE", "QUANTIDADE_PRODUTO", "MAQUINAS_NECESSARIAS", "DATA_INICIO", "TEMPO_PARADA_MAQUINA_MIN", "QTD_REFUGADA"]
OPTIONAL = ["TURNOS_NECESSARIOS", "QUANTIDADE_PLANEJADA"]

def norm(s):
    if not s: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def parse_number(x):
    if pd.isna(x): return pd.NA
    return float(str(x).replace(".", "").replace(",", ".")) if re.search(r"\d", str(x)) else pd.NA

def parse_date_ptbr(x):
    if pd.isna(x): return pd.NaT
    try: return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except: return pd.NaT

@st.cache_data
def try_read_csv_bytes(content: bytes):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if df.shape[1] > 1: return df
        except: continue
    return pd.read_csv(io.BytesIO(content))

def auto_map(df):
    mapping = {}
    for k, aliases in ALIASES.items():
        found = next((c for c in df.columns if norm(c) in [norm(a) for a in aliases]), None)
        mapping[k] = found
    return mapping

def build_clean(df, mapping):
    clean = pd.DataFrame()
    for key in REQUIRED + OPTIONAL:
        if mapping.get(key) and mapping[key] in df.columns:
            clean[key] = df[mapping[key]]
    if "DATA_INICIO" in clean: clean["DATA_INICIO"] = clean["DATA_INICIO"].apply(parse_date_ptbr)
    for numcol in ["QUANTIDADE_PRODUTO", "TEMPO_PARADA_MAQUINA_MIN", "QTD_REFUGADA"]:
        if numcol in clean: clean[numcol] = clean[numcol].apply(parse_number)
    return clean

def compute_oee(df):
    df = df.copy()
    tempo_planejado = df.get("TURNOS_NECESSARIOS", pd.Series(1, index=df.index)) * 8 * 60
    tempo_operacao = tempo_planejado - df["TEMPO_PARADA_MAQUINA_MIN"].fillna(0)
    disponibilidade = (tempo_operacao / tempo_planejado).clip(0, 1)
    produzido = df["QUANTIDADE_PRODUTO"].fillna(0)
    planejado = df.get("QUANTIDADE_PLANEJADA", produzido)
    performance = (produzido / planejado).replace([pd.NA, float("inf")], 0).clip(0, 2)
    qualidade = ((produzido - df["QTD_REFUGADA"].fillna(0)) / produzido.replace(0, pd.NA)).fillna(1).clip(0, 1)
    oee = disponibilidade * performance * qualidade
    df["DISPONIBILIDADE"] = disponibilidade
    df["PERFORMANCE"] = performance
    df["QUALIDADE"] = qualidade
    df["OEE"] = oee
    return df

# ============================================================
# HERO IMAGE
# ============================================================
if hero_image:
    st.image(hero_image, use_container_width=True)
else:
    st.markdown(
        '<div class="hero"><h1>Easy Textil — Seu medidor de eficiência</h1><div class="sub">OEE = Disponibilidade × Performance × Qualidade</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📄 Pré-visualização", "🧭 Mapeamento", "🧼 Dados limpos", "📊 OEE"])

# -------------------------- TAB 1 ----------------------------
if uploaded_csv:
    df_raw = try_read_csv_bytes(uploaded_csv.read())
    with tab1:
        st.dataframe(df_raw.head(20), use_container_width=True)
else:
    with tab1:
        st.info("👈 Envie um CSV na barra lateral.")
        st.stop()

# -------------------------- TAB 2 ----------------------------
with tab2:
    auto = auto_map(df_raw)
    st.subheader("🧭 Confirme/ajuste o mapeamento")
    cols = list(df_raw.columns)
    pick = {}
    grid = st.columns(3)
    for i, key in enumerate(REQUIRED + OPTIONAL):
        pick[key] = grid[i % 3].selectbox(key, ["—"] + cols, index=(cols.index(auto[key]) + 1) if auto[key] in cols else 0)
    if st.button("✅ Aplicar mapeamento"):
        st.session_state.mapping_manual = {k: (None if v == "—" else v) for k, v in pick.items()}
        st.success("Mapeamento salvo!")

# -------------------------- TAB 3 ----------------------------
final_map = st.session_state.get("mapping_manual", auto_map(df_raw))
clean_df = build_clean(df_raw, final_map)

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", len(clean_df))
    c2.metric("Refugo total", float(clean_df["QTD_REFUGADA"].fillna(0).sum()))
    c3.metric("Média parada (min)", round(clean_df["TEMPO_PARADA_MAQUINA_MIN"].mean(), 1))
    st.dataframe(clean_df.head(50), use_container_width=True)

# -------------------------- TAB 4 ----------------------------
with tab4:
    oee_df = compute_oee(clean_df)
    d_mean = oee_df["DISPONIBILIDADE"].mean()
    p_mean = oee_df["PERFORMANCE"].mean()
    q_mean = oee_df["QUALIDADE"].mean()
    o_mean = oee_df["OEE"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disponibilidade", f"{d_mean:.1%}")
    c2.metric("Performance", f"{p_mean:.1%}")
    c3.metric("Qualidade", f"{q_mean:.1%}")
    c4.metric("OEE Médio", f"{o_mean:.1%}")

    st.line_chart(oee_df[["OEE"]])
    st.dataframe(oee_df, use_container_width=True)
