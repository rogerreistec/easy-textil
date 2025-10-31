# app.py
# Easy Textil — Carregador/normalizador de CSV com logo fixa em assets/logo.jpg
# Mantém tudo igual, apenas garantindo a exibição da logo e ícone.

import io
import re
import os
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# LOGO fixa em assets/logo.jpg (usada também como favicon)
# ============================================================
LOGO_PATH = "assets/logo.jpg"

if not os.path.exists(LOGO_PATH):
    raise FileNotFoundError(
        "❌ Logo não encontrada! Coloque o arquivo 'logo.jpg' dentro da pasta /assets."
    )

logo_image = Image.open(LOGO_PATH)

# ============================================================
# Configuração da página com favicon da logo
# ============================================================
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon=logo_image,
    layout="wide",
)

# ============================================================
# Cabeçalho com logo e título
# ============================================================
def show_header():
    left, right = st.columns([1, 3])
    with left:
        st.image(logo_image, caption=None, use_container_width=True)
    with right:
        st.title("Easy Textil — Seu medidor de eficiência")
        st.caption("OEE = **Disponibilidade × Performance × Qualidade**")

show_header()

# ============================================================
# Logo também na barra lateral
# ============================================================
st.sidebar.image(logo_image, use_container_width=True)
st.sidebar.caption("Seu medidor de eficiência")

# ============================================================
# Passo a passo para leigos (UI)
# ============================================================
with st.expander("📋 Passo a passo (bem simples) — clique para ver"):
    st.markdown(
        """
        1. **Baixe o CSV Modelo** (na barra lateral) e veja como os campos ficam.
        2. **Arraste seu CSV** para a barra lateral ou clique em **Carregar**.
        3. **Confirme o mapeamento** das colunas obrigatórias (na barra lateral).
        4. Veja os **dados limpos** e, se quiser, **baixe o CSV Limpo**.
        5. (Opcional) Veja as **métricas rápidas** no final.
        """
    )

with st.expander("🧠 Como preparar seu CSV (exemplos práticos)"):
    st.markdown(
        """
        - **Datas** podem estar como `12-dez.`, `09/jan.`, `7-jan.`, `23/jan`, `23-jan-24`, `24/01/2025` — eu reconheço todas.
        - **Números**: `11.500` (milhar) e `23,8` (decimal) são entendidos automaticamente.
        - **CLIENTE dentro de PEDIDO**: se vier `98406 JK INDUSTRIA`, eu extraio **JK INDUSTRIA**.
        - **Parada (minutos)**: se tiver apenas **HORAS MAQ.**, eu converto para **minutos**.
        - **Máquinas necessárias**: se não houver, uso **TURNOS NECESSÁRIOS**; se não houver nenhum, assumo **1**.
        """
    )

# ============================================================
# Colunas obrigatórias do pipeline
# ============================================================
REQUIRED = [
    "CLIENTE",
    "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS",
    "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN",
    "QTD_REFUGADA",
]

ALIASES: Dict[str, List[str]] = {
    "CLIENTE": ["cliente", "pedido", "cliente/pedido"],
    "QUANTIDADE_PRODUTO": ["metros", "quantidade", "kg prod.", "kg produzido", "kg"],
    "MAQUINAS_NECESSARIAS": ["turnos", "maq.", "maquinas"],
    "DATA_INICIO": ["inicio prod.", "data inicio"],
    "TEMPO_PARADA_MAQUINA_MIN": ["horas maq.", "parada"],
    "QTD_REFUGADA": ["kg rest.", "refugo"],
}

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12
}

# ============================================================
# Funções utilitárias
# ============================================================
def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def parse_number(x) -> Optional[float]:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return pd.NA

def parse_date_ptbr(x) -> Optional[pd.Timestamp]:
    if pd.isna(x) or str(x).strip() == "" or str(x).strip() == "**":
        return pd.NaT
    s = str(x).strip()
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt
    except Exception:
        pass
    m = re.match(r"(\d{1,2})[-/ ]([A-Za-z]{3,})[-/ ]?(\d{2,4})?", s)
    if m:
        d = int(m.group(1))
        mon = PT_MONTHS.get(m.group(2)[:3].lower())
        y = int(m.group(3)) + 2000 if m.group(3) and len(m.group(3)) == 2 else int(m.group(3) or datetime.now().year)
        try:
            return pd.Timestamp(year=y, month=mon, day=d)
        except Exception:
            return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def extract_cliente_from_pedido(series: pd.Series) -> pd.Series:
    def _extract(s):
        if pd.isna(s):
            return pd.NA
        m = re.match(r"^\s*\d+\s+(.*)$", str(s))
        return m.group(1).strip() if m else str(s)
    return series.apply(_extract)

def try_read_csv(file) -> pd.DataFrame:
    content = file.read()
    for sep in [",", ";", "\t"]:
        for enc in ["utf-8-sig", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(content))

# ============================================================
# Upload do CSV
# ============================================================
uploaded = st.sidebar.file_uploader("📥 Carregue um CSV", type=["csv"])

MODEL_DF = pd.DataFrame(
    {
        "CLIENTE": ["Exemplo Indústria"],
        "QUANTIDADE_PRODUTO": [1000],
        "MAQUINAS_NECESSARIAS": [1],
        "DATA_INICIO": ["2025-01-07"],
        "TEMPO_PARADA_MAQUINA_MIN": [0],
        "QTD_REFUGADA": [0],
    }
)
st.sidebar.download_button(
    "⬇️ Baixar CSV Modelo",
    data=MODEL_DF.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_modelo.csv",
)

if uploaded is None:
    st.info("Carregue um arquivo CSV na barra lateral.")
    st.stop()

# ============================================================
# Leitura e processamento do CSV
# ============================================================
df = try_read_csv(uploaded)
st.subheader("📄 Pré-visualização do CSV")
st.dataframe(df.head(15), use_container_width=True)

# ============================================================
# Mapeamento automático e ajustes
# ============================================================
cols = list(df.columns)
mapping = {}

for key, aliases in ALIASES.items():
    for col in cols:
        if norm(col) in [norm(a) for a in aliases]:
            mapping[key] = col
            break
    else:
        mapping[key] = None

clean = pd.DataFrame()

# Cliente
if mapping["CLIENTE"]:
    clean["CLIENTE"] = df[mapping["CLIENTE"]].astype(str)
else:
    ped_col = next((c for c in cols if "pedido" in norm(c)), None)
    clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]) if ped_col else pd.NA

# Quantidade
if mapping["QUANTIDADE_PRODUTO"]:
    clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
else:
    qtd_col = next((c for c in cols if "metro" in norm(c)), None)
    clean["QUANTIDADE_PRODUTO"] = df[qtd_col].apply(parse_number) if qtd_col else pd.NA

# Máquinas necessárias
if mapping["MAQUINAS_NECESSARIAS"]:
    clean["MAQUINAS_NECESSARIAS"] = df[mapping["MAQUINAS_NECESSARIAS"]].apply(parse_number).fillna(1).astype("Int64")
else:
    clean["MAQUINAS_NECESSARIAS"] = 1

# Data início
if mapping["DATA_INICIO"]:
    clean["DATA_INICIO"] = df[mapping["DATA_INICIO"]].apply(parse_date_ptbr)
else:
    clean["DATA_INICIO"] = pd.NaT

# Tempo parada
if mapping["TEMPO_PARADA_MAQUINA_MIN"]:
    clean["TEMPO_PARADA_MAQUINA_MIN"] = df[mapping["TEMPO_PARADA_MAQUINA_MIN"]].apply(parse_number)
else:
    clean["TEMPO_PARADA_MAQUINA_MIN"] = 0

# Refugada
if mapping["QTD_REFUGADA"]:
    clean["QTD_REFUGADA"] = df[mapping["QTD_REFUGADA"]].apply(parse_number)
else:
    clean["QTD_REFUGADA"] = 0

# ============================================================
# Exibição final
# ============================================================
st.subheader("🧼 Dados limpos")
st.dataframe(clean.head(50), use_container_width=True)

st.download_button(
    "⬇️ Baixar CSV LIMPO",
    data=clean.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_limpo.csv",
)
