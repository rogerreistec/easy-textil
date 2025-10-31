# app.py — Easy Textil (Seu medidor de eficiência)
# Streamlit 1.50+
from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Aparência básica
# ---------------------------------------------------------
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1F4B99"

# ---------------------------------------------------------
# Utilitários: logo e csv modelo
# ---------------------------------------------------------
def show_logo():
    logo_file = Path("assets/logo.jpg")
    if logo_file.exists():
        st.sidebar.image(str(logo_file), use_container_width=True)
    else:
        st.sidebar.warning("⚠️ Coloque sua logo em **assets/logo.jpg** para exibir aqui.")


def build_template_csv() -> bytes:
    """Gera um CSV vazio com cabeçalhos canônicos aceitos pelo app."""
    cols = [
        "PRODUTO",
        "CLIENTE",
        "QUANTIDADE_PRODUTO",
        "MAQUINAS_NECESSARIAS",
        "DATA_INICIO",                 # dd/mm/aaaa hh:mm (tolerante)
        "TEMPO_PARADA_MAQUINA_MIN",    # número em minutos
        "QTD_REFUGADA",                # peças
        # opcionais, mas ajudam muito:
        "QUANTIDADE_PRODUZIDA",        # peças boas + refugadas (se omitir, o app calcula)
        "TEMPO_PLANEJADO_MIN",         # se omitir, o app infere do ciclo ideal
        "DATA_FIM",
        # perdas detalhadas (opcionais)
        "PARADA_QUEBRA_MIN",
        "PARADA_SETUP_AJUSTE_MIN",
        "MICROPARADAS_MIN",
        "RENDIMENTO_REFUGO_QTD",
        "DEFEITOS_RETRABALHO_QTD",
    ]
    df = pd.DataFrame(columns=cols)
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")


# ---------------------------------------------------------
# Normalização de cabeçalhos e leitura flexível
# ---------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


# sinônimos PT/BR comuns -> nome canônico
CANON_MAP: Dict[str, str] = {
    # obrigatórias
    "produto": "PRODUTO",
    "artigo": "PRODUTO",
    "cliente": "CLIENTE",
    "quantidade_produto": "QUANTIDADE_PRODUTO",
    "quantidade": "QUANTIDADE_PRODUTO",
    "qtd": "QUANTIDADE_PRODUTO",
    "maquinas_necessarias": "MAQUINAS_NECESSARIAS",
    "maquina": "MAQUINAS_NECESSARIAS",
    "maquinas": "MAQUINAS_NECESSARIAS",
    "data_inicio": "DATA_INICIO",
    "inicio": "DATA_INICIO",
    "dt_inicio": "DATA_INICIO",
    "tempo_parada_maquina_min": "TEMPO_PARADA_MAQUINA_MIN",
    "tempo_parada": "TEMPO_PARADA_MAQUINA_MIN",
    "paradas_min": "TEMPO_PARADA_MAQUINA_MIN",
    "qtd_refugada": "QTD_REFUGADA",
    "refugo": "QTD_REFUGADA",
    "refugos": "QTD_REFUGADA",
    "quantidade_refugada": "QTD_REFUGADA",
    # opcionais
    "quantidade_produzida": "QUANTIDADE_PRODUZIDA",
    "producao_real": "QUANTIDADE_PRODUZIDA",
    "tempo_planejado_min": "TEMPO_PLANEJADO_MIN",
    "tempo_planejado": "TEMPO_PLANEJADO_MIN",
    "data_fim": "DATA_FIM",
    "dt_fim": "DATA_FIM",
    # perdas detalhadas
    "parada_quebra_min": "PARADA_QUEBRA_MIN",
    "parada_setup_ajuste_min": "PARADA_SETUP_AJUSTE_MIN",
    "setup_ajuste_min": "PARADA_SETUP_AJUSTE_MIN",
    "microparadas_min": "MICROPARADAS_MIN",
    "rendimento_refugo_qtd": "RENDIMENTO_REFUGO_QTD",
    "defeitos_retrabalho_qtd": "DEFEITOS_RETRABALHO_QTD",
}

REQUIRED = [
    "PRODUTO",
    "CLIENTE",
    "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS",
    "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN",
    "QTD_REFUGADA",
]


def read_csv_flexible(file) -> pd.DataFrame:
    # detecta ; ou , automaticamente
    df = pd.read_csv(file, sep=None, engine="python", decimal=",")
    # normaliza e mapeia
    normalized = [_norm(c) for c in df.columns]
    mapped = [CANON_MAP.get(n, n.upper()) for n in normalized]
    df.columns = mapped

    # confirma obrigatórias
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing) +
            ". Dica: baixe o CSV modelo na barra lateral."
        )

    # datas
    df["DATA_INICIO"] = pd.to_datetime(df["DATA_INICIO"], dayfirst=True, errors="coerce")
    if "DATA_FIM" in df.columns:
        df["DATA_FIM"] = pd.to_datetime(df.get("DATA_FIM"), dayfirst=True, errors="coerce")

    # numéricos
    for col in ["QUANTIDADE_PRODUTO", "TEMPO_PARADA_MAQUINA_MIN", "QTD_REFUGADA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "QUANTIDADE_PRODUZIDA" in df.columns:
        df["QUANTIDADE_PRODUZIDA"] = pd.to_numeric(df["QUANTIDADE_PRODUZIDA"], errors="coerce").fillna(0)

    if "TEMPO_PLANEJADO_MIN" in df.columns:
        df["TEMPO_PLANEJADO_MIN"] = pd.to_numeric(df["TEMPO_PLANEJADO_MIN"], errors="coerce").fillna(0)

    # perdas detalhadas opcionais
    for col in [
        "PARADA_QUEBRA_MIN",
        "PARADA_SETUP_AJUSTE_MIN",
        "MICROPARADAS_MIN",
        "RENDIMENTO_REFUGO_QTD",
        "DEFEITOS_RETRABALHO_QTD",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# ---------------------------------------------------------
# Cálculos OEE
# ---------------------------------------------------------
def infer_tempo_planejado(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Se TEMPO_PLANEJADO_MIN existir, usa. Se não:
    - Se houver DATA_FIM, usa diferença DATA_FIM - DATA_INICIO.
    - Senão, tenta inferir tempo de ciclo ideal = (tempo_planejado/operação) a partir de
      QUANTIDADE_PRODUZIDA ou QUANTIDADE_PRODUTO.
    """
    msg = ""
    if "TEMPO_PLANEJADO_MIN" in df.columns and df["TEMPO_PLANEJADO_MIN"].sum() > 0:
        return df, "Tempo planejado lido do CSV."

    if "DATA_FIM" in df.columns and df["DATA_FIM"].notna().any():
        mins = (df["DATA_FIM"] - df["DATA_INICIO"]).dt.total_seconds() / 60.0
        df["TEMPO_PLANEJADO_MIN"] = pd.to_numeric(mins, errors="coerce").fillna(0)
        return df, "Tempo planejado inferido por DATA_INICIO/DATA_FIM."

    # inferência simples pelo ciclo ideal aproximado
    produced = df.get("QUANTIDADE_PRODUZIDA")
    if produced is None or produced.sum() == 0:
        produced = df["QUANTIDADE_PRODUTO"].fillna(0)

    # chute conservador: se não há referência, considera 1,0 min/peça
    # (o usuário pode ajustar depois incluindo TEMPO_PLANEJADO_MIN no CSV)
    ciclo_ideal_min = 1.0
    df["TEMPO_PLANEJADO_MIN"] = (produced * ciclo_ideal_min).astype(float)
    msg = "Tempo planejado não informado; inferido por ciclo ideal padrão (1,0 min/peça)."
    return df, msg


def compute_oee(df: pd.DataFrame) -> pd.DataFrame:
    # QUANTIDADE_PRODUZIDA: se não tiver, usamos QUANTIDADE_PRODUTO (pedido) como aproximado
    if "QUANTIDADE_PRODUZIDA" not in df.columns or (df["QUANTIDADE_PRODUZIDA"].sum() == 0):
        df["QUANTIDADE_PRODUZIDA"] = pd.to_numeric(df["QUANTIDADE_PRODUTO"], errors="coerce").fillna(0)

    # tempo planejado
    df, tp_msg = infer_tempo_planejado(df)

    # tempos
    df["TEMPO_DISPONIVEL_MIN"] = df["TEMPO_PLANEJADO_MIN"].clip(lower=0)
    df["TEMPO_OPERACAO_MIN"] = (df["TEMPO_DISPONIVEL_MIN"] - df["TEMPO_PARADA_MAQUINA_MIN"]).clip(lower=0)

    # ciclo ideal (min/peça)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["CICLO_IDEAL_MIN_POR_PECA"] = np.where(
            df["QUANTIDADE_PRODUZIDA"] > 0,
            df["TEMPO_DISPONIVEL_MIN"] / df["QUANTIDADE_PRODUZIDA"],
            np.nan,
        )

    # componentes OEE
    dispon = np.where(
        df["TEMPO_DISPONIVEL_MIN"] > 0,
        (df["TEMPO_DISPONIVEL_MIN"] - df["TEMPO_PARADA_MAQUINA_MIN"]) / df["TEMPO_DISPONIVEL_MIN"],
        0,
    )
    df["DISPONIBILIDADE"] = np.clip(dispon, 0, 1)

    # Performance = Produção Real × Tempo de Ciclo Ideal / Tempo de Operação
    ciclo_ideal = df["CICLO_IDEAL_MIN_POR_PECA"].fillna(0)
    oper = df["TEMPO_OPERACAO_MIN"].replace(0, np.nan)
    perf = (df["QUANTIDADE_PRODUZIDA"] * ciclo_ideal) / oper
    perf = perf.replace([np.inf, -np.inf, np.nan], 0.0)
    df["PERFORMANCE"] = np.clip(perf, 0, 1)

    # Qualidade = (Produzida - Refugo)/Produzida
    prod = df["QUANTIDADE_PRODUZIDA"].replace(0, np.nan)
    qual = (df["QUANTIDADE_PRODUZIDA"] - df["QTD_REFUGADA"]) / prod
    qual = qual.replace([np.inf, -np.inf, np.nan], 0.0)
    df["QUALIDADE"] = np.clip(qual, 0, 1)

    df["OEE"] = df["DISPONIBILIDADE"] * df["PERFORMANCE"] * df["QUALIDADE"]

    # arredonda para leitura
    for c in ["DISPONIBILIDADE", "PERFORMANCE", "QUALIDADE", "OEE"]:
        df[c] = (df[c] * 100).round(2)

    df["_tp_msg"] = tp_msg
    return df


# ---------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------
show_logo()
st.sidebar.title("Easy Textil")
st.sidebar.caption("Seu medidor de eficiência")

uploaded = st.sidebar.file_uploader("📥 Carregue um CSV", type=["csv"])
st.sidebar.download_button(
    "Baixar CSV modelo",
    data=build_template_csv(),
    file_name="easy_textil_modelo.csv",
    mime="text/csv",
    help="Use este cabeçalho como referência.",
)

st.sidebar.markdown("---")
font_size = st.sidebar.slider("Acessibilidade — Tamanho do texto", 16, 24, 18)
st.markdown(
    f"""
    <style>
    html, body, [class*="css"]  {{
        font-size: {font_size}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# UI — Header
# ---------------------------------------------------------
st.markdown(
    f"""
    <h1 style="margin-bottom:0">Easy Textil — Seu medidor de eficiência</h1>
    <p style="color:#555;margin-top:4px">OEE = <b>Disponibilidade × Performance × Qualidade</b></p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Fluxo principal
# ---------------------------------------------------------
if not uploaded:
    st.info(
        "Carregue um **CSV** na barra lateral para visualizar os indicadores. "
        "Se precisar, baixe o **CSV modelo**."
    )
    st.stop()

# tenta ler com robustez
try:
    df_raw = read_csv_flexible(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# calcula OEE
df = compute_oee(df_raw.copy())

# mensagem sobre tempo planejado
if df["_tp_msg"].iloc[0]:
    st.warning(df["_tp_msg"].iloc[0])

# ---------------------------------------------------------
# KPIs gerais
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
dispon_m = df["DISPONIBILIDADE"].mean()
perf_m = df["PERFORMANCE"].mean()
qual_m = df["QUALIDADE"].mean()
oee_m = df["OEE"].mean()

col1.metric("Disponibilidade média", f"{dispon_m:.1f}%")
col2.metric("Performance média", f"{perf_m:.1f}%")
col3.metric("Qualidade média", f"{qual_m:.1f}%")
col4.metric("OEE médio", f"{oee_m:.1f}%")

st.markdown("---")

# ---------------------------------------------------------
# Tabelas e gráficos
# ---------------------------------------------------------
with st.expander("Ver tabela base (dados normalizados)", expanded=False):
    st.dataframe(df.drop(columns=["_tp_msg"]), use_container_width=True)

# OEE por máquina
if "MAQUINAS_NECESSARIAS" in df.columns:
    oee_maquina = (
        df.groupby("MAQUINAS_NECESSARIAS", dropna=False)["OEE"]
        .mean()
        .reset_index()
        .sort_values("OEE", ascending=False)
    )
    fig = px.bar(
        oee_maquina,
        x="MAQUINAS_NECESSARIAS",
        y="OEE",
        title="OEE por Máquina",
        labels={"MAQUINAS_NECESSARIAS": "Máquina", "OEE": "OEE (%)"},
    )
    fig.update_layout(yaxis=dict(range=[0, 100]), height=420)
    st.plotly_chart(fig, use_container_width=True)

# Perdas detalhadas (se existirem)
loss_cols = [c for c in ["PARADA_QUEBRA_MIN", "PARADA_SETUP_AJUSTE_MIN", "MICROPARADAS_MIN"] if c in df.columns]
if loss_cols:
    loss = df[loss_cols].sum().sort_values(ascending=False).reset_index()
    loss.columns = ["Tipo de perda (min)", "Minutos"]
    fig2 = px.pie(loss, values="Minutos", names="Tipo de perda (min)", title="Distribuição de perdas (min)")
    st.plotly_chart(fig2, use_container_width=True)

# Mapa de Pareto simples de refugo
if "QTD_REFUGADA" in df.columns and "PRODUTO" in df.columns:
    pareto = (
        df.groupby("PRODUTO", dropna=False)["QTD_REFUGADA"]
        .sum()
        .reset_index()
        .sort_values("QTD_REFUGADA", ascending=False)
        .head(15)
    )
    fig3 = px.bar(
        pareto,
        x="PRODUTO",
        y="QTD_REFUGADA",
        title="Top 15 Refugos por Produto",
        labels={"QTD_REFUGADA": "Refugo (peças)"},
    )
    st.plotly_chart(fig3, use_container_width=True)

st.caption("© Easy Textil — OEE para a indústria têxtil, simples e direto.")
