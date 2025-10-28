# -*- coding: utf-8 -*-
"""
Easy O.E.E. Têxtil — MVP acessível e direto ao ponto
- Upload de CSV
- Cálculo de OEE (Disponibilidade, Performance, Qualidade)
- Perdas por pilares (têxtil)
- Gráficos (Plotly) e exportações (CSV/Excel)
Compatível com Streamlit 1.50+
"""

import io
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Configuração de página e acessibilidade
# ------------------------------------------------------------
st.set_page_config(page_title="Easy O.E.E. Têxtil", page_icon="🧵", layout="wide")

ACCESS_CSS = """
<style>
:root { --base-font-size: 18px; }
html, body, [class*="css"]  { font-size: var(--base-font-size) !important; }
section[data-testid="stSidebar"] { font-size: 18px; }
button, .stButton button { font-size: 18px !important; padding: 0.6rem 1rem; }
label, .stMarkdown p { line-height: 1.5; }
[data-testid="stMetricValue"] { font-size: 36px !important; }
</style>
"""
st.markdown(ACCESS_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------
with st.sidebar:
    st.header("1) Carregar CSV")
    file = st.file_uploader("Selecione um arquivo .csv", type=["csv"])

    st.header("2) Acessibilidade")
    base_font = st.slider("Tamanho do texto", 16, 24, 18)
    st.markdown(f"<style>:root {{ --base-font-size: {base_font}px; }}</style>", unsafe_allow_html=True)

    st.header("3) Modelo (opcional)")
    st.caption("Baixe um CSV de exemplo com as colunas esperadas.")
    if st.button("⬇️ Baixar modelo (.csv)"):
        exemplo = pd.DataFrame({
            "PRODUTO": ["Tecido A", "Tecido B"],
            "CLIENTE": ["Cliente 1", "Cliente 2"],
            "QUANTIDADE_PRODUTO": [1000, 800],
            "QUANTIDADE_PRODUZIDA": [980, 760],  # opcional
            "MAQUINAS_NECESSARIAS": ["Urdideira; Tear", "Tear"],
            "DATA_INICIO": ["2025-09-01 08:00", "2025-09-02 08:00"],
            "TEMPO_PLANEJADO_MIN": [600, 480],          # minutos
            "TEMPO_PARADA_MAQUINA_MIN": [60, 40],       # minutos
            "QTD_REFUGADA": [15, 20],
            # Perdas por pilar (opcionais)
            "PARADA_QUEBRA_MIN": [30, 20],
            "PARADA_SETUP_AJUSTE_MIN": [30, 20],
            "MICROPARADAS_MIN": [10, 8],
            "RENDIMENTO_REFUGO_QTD": [5, 4],
            "DEFEITOS_RETRABALHO_QTD": [10, 16],
        })
        st.download_button(
            "Baixar modelo",
            exemplo.to_csv(index=False).encode("utf-8"),
            file_name="exemplo_easy_oee_textil.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------
# Regras de colunas
# ------------------------------------------------------------
REQ_BASE = [
    "PRODUTO", "CLIENTE", "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS", "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN", "QTD_REFUGADA"
]
OPT = [
    "TEMPO_PLANEJADO_MIN", "TEMPO_PLANEJADO",
    "QUANTIDADE_PRODUZIDA", "DATA_FIM",
    "PARADA_QUEBRA_MIN", "PARADA_SETUP_AJUSTE_MIN",
    "MICROPARADAS_MIN", "RENDIMENTO_REFUGO_QTD",
    "DEFEITOS_RETRABALHO_QTD"
]

# ------------------------------------------------------------
# Leitura e validação do CSV
# ------------------------------------------------------------
def carregar_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    # checa base obrigatória
    faltando = [c for c in REQ_BASE if c not in df.columns]
    if faltando:
        st.error(f"Colunas obrigatórias ausentes: {', '.join(faltando)}")
        st.stop()

    # opcionais padrão
    for c in OPT:
        if c not in df.columns:
            if c == "DATA_FIM":
                df[c] = pd.NaT
            else:
                df[c] = 0

    # Normaliza TEMPO_PLANEJADO_MIN (aceita TEMPO_PLANEJADO)
    if "TEMPO_PLANEJADO_MIN" not in df.columns or (pd.to_numeric(df["TEMPO_PLANEJADO_MIN"], errors="coerce").fillna(0) == 0).all():
        if "TEMPO_PLANEJADO" in df.columns:
            df["TEMPO_PLANEJADO_MIN"] = pd.to_numeric(df["TEMPO_PLANEJADO"], errors="coerce").fillna(0)
        else:
            st.error("Informe 'TEMPO_PLANEJADO_MIN' (min) ou 'TEMPO_PLANEJADO' (min) no CSV.")
            st.stop()

    # Tipos numéricos
    num_cols = [
        "QUANTIDADE_PRODUTO", "QUANTIDADE_PRODUZIDA", "TEMPO_PLANEJADO_MIN",
        "TEMPO_PARADA_MAQUINA_MIN", "QTD_REFUGADA",
        "PARADA_QUEBRA_MIN", "PARADA_SETUP_AJUSTE_MIN",
        "MICROPARADAS_MIN", "RENDIMENTO_REFUGO_QTD",
        "DEFEITOS_RETRABALHO_QTD"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Datas
    df["DATA_INICIO"] = pd.to_datetime(df["DATA_INICIO"], errors="coerce")
    if "DATA_FIM" in df.columns:
        df["DATA_FIM"] = pd.to_datetime(df["DATA_FIM"], errors="coerce")

    return df

# ------------------------------------------------------------
# Cálculo do OEE e perdas
# ------------------------------------------------------------
def calcular_oee(df: pd.DataFrame) -> pd.DataFrame:
    # Produção real (se não houver, usa a planejada)
    df["QUANTIDADE_PRODUZIDA"] = pd.to_numeric(df.get("QUANTIDADE_PRODUZIDA", 0), errors="coerce").fillna(0)
    df["PRODUCAO_REAL"] = df["QUANTIDADE_PRODUZIDA"].where(df["QUANTIDADE_PRODUZIDA"] > 0, df["QUANTIDADE_PRODUTO"])

    # Tempo de produção real
    df["TEMPO_PRODUCAO_REAL_MIN"] = (df["TEMPO_PLANEJADO_MIN"] - df["TEMPO_PARADA_MAQUINA_MIN"]).clip(lower=0)

    # Disponibilidade = (Tempo Planejado - Parada) / Tempo Planejado
    disp_denom = df["TEMPO_PLANEJADO_MIN"].replace(0, pd.NA)
    df["Disponibilidade"] = (df["TEMPO_PRODUCAO_REAL_MIN"] / disp_denom).fillna(0).clip(0, 1)

    # Tempo de ciclo ideal (min/unid)
    ciclo_ideal = (df["TEMPO_PLANEJADO_MIN"] / df["QUANTIDADE_PRODUTO"].replace(0, pd.NA)).fillna(0)
    df["TEMPO_CICLO_IDEAL_MIN"] = ciclo_ideal

    # Performance = Produção Real * Ciclo Ideal / Tempo Produção Real
    perf_denom = df["TEMPO_PRODUCAO_REAL_MIN"].replace(0, pd.NA)
    df["Performance"] = (df["PRODUCAO_REAL"] * df["TEMPO_CICLO_IDEAL_MIN"] / perf_denom).fillna(0).clip(0, 1)

    # Qualidade = (Produção Real - Refugo) / Produção Real
    q_denom = df["PRODUCAO_REAL"].replace(0, pd.NA)
    df["Qualidade"] = ((df["PRODUCAO_REAL"] - df["QTD_REFUGADA"]) / q_denom).fillna(0).clip(0, 1)

    # OEE
    df["OEE"] = (df["Disponibilidade"] * df["Performance"] * df["Qualidade"]).fillna(0)

    # DATA_FIM automática, se faltar
    if "DATA_FIM" in df.columns:
        mask = df["DATA_FIM"].isna() & df["DATA_INICIO"].notna()
        df.loc[mask, "DATA_FIM"] = df.loc[mask, "DATA_INICIO"] + pd.to_timedelta(df.loc[mask, "TEMPO_PLANEJADO_MIN"], unit="m")

    # Perdas por pilar
    for c in ["PARADA_QUEBRA_MIN", "PARADA_SETUP_AJUSTE_MIN", "MICROPARADAS_MIN", "RENDIMENTO_REFUGO_QTD", "DEFEITOS_RETRABALHO_QTD"]:
        if c not in df.columns:
            df[c] = 0

    df["PERDA_DISPON_MIN"] = (df["PARADA_QUEBRA_MIN"] + df["PARADA_SETUP_AJUSTE_MIN"]).fillna(0)  # minutos
    df["PERDA_PERF_MIN"]   = df["MICROPARADAS_MIN"].fillna(0)                                      # minutos
    df["PERDA_QUAL_QTD"]   = (df["RENDIMENTO_REFUGO_QTD"] + df["DEFEITOS_RETRABALHO_QTD"]).fillna(0)  # unidades

    return df

def kpi(label, value):
    box = st.container(border=True)
    with box:
        st.markdown(f"### {label}")
        st.markdown(f"<div style='font-size:36px;font-weight:700;'>{value:.1%}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Interface principal
# ------------------------------------------------------------
st.title("🧵 Easy O.E.E. Têxtil")

if not file:
    st.info("👉 Envie um **CSV** na barra lateral para visualizar o painel. Você pode baixar um modelo.")
    st.stop()

df_raw = carregar_csv(file)
df = calcular_oee(df_raw.copy())

# KPIs
st.subheader("📌 Indicadores principais (média)")
c1, c2, c3, c4 = st.columns(4)
with c1: kpi("OEE", df["OEE"].mean())
with c2: kpi("Disponibilidade", df["Disponibilidade"].mean())
with c3: kpi("Performance", df["Performance"].mean())
with c4: kpi("Qualidade", df["Qualidade"].mean())

st.divider()

# Tabela
st.subheader("🔎 Tabela de ordens (com cálculos)")
st.dataframe(df.head(300), use_container_width=True)

# Gráfico OEE por Produto
st.subheader("📊 OEE por Produto")
g_prod = df.groupby("PRODUTO", as_index=False)["OEE"].mean().sort_values("OEE", ascending=False)
fig1 = px.bar(g_prod, x="PRODUTO", y="OEE", text="OEE", title="OEE médio por produto")
fig1.update_traces(texttemplate="%{text:.1%}", textposition="outside")
fig1.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig1, use_container_width=True)

# Perdas por Pilar
st.subheader("🧭 Perdas por Pilar (médias)")
pilar = pd.DataFrame({
    "Pilar": ["Disponibilidade (min)", "Performance (min)", "Qualidade (unid)"],
    "Valor": [df["PERDA_DISPON_MIN"].mean(), df["PERDA_PERF_MIN"].mean(), df["PERDA_QUAL_QTD"].mean()]
})
fig2 = px.bar(pilar, x="Pilar", y="Valor", text="Valor", title="Perdas médias por pilar")
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

# Exportações
st.subheader("📤 Exportar resultados")
colA, colB = st.columns(2)
with colA:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV com resultados", csv_bytes, file_name="easy_oee_textil_resultados.csv", mime="text/csv")
with colB:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wb:
        df.to_excel(wb, index=False, sheet_name="Resultados")
        g_prod.to_excel(wb, index=False, sheet_name="OEE_por_Produto")
        pilar.to_excel(wb, index=False, sheet_name="Perdas_Pilares")
    st.download_button(
        "Baixar Excel (múltiplas abas)",
        out.getvalue(),
        file_name="easy_oee_textil_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.caption("Dica: títulos claros, poucos gráficos por tela e fonte maior ajudam na leitura.")
