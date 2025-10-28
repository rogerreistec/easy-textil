# -*- coding: utf-8 -*-
"""
Easy Textil Code — Seu medidor de eficiência
Propósito: medir OEE (Disponibilidade x Performance x Qualidade) com foco em acessibilidade.
"""

from datetime import datetime
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Easy Textil Code — Seu medidor de eficiência", page_icon="🧵", layout="wide")

BRAND_CSS = """
<style>
:root { --base-font-size: 18px; }
html, body, [class*="css"] { font-size: var(--base-font-size) !important; }
section[data-testid="stSidebar"] { font-size: 18px; }
.stButton>button { font-size: 18px !important; padding:.55rem 1rem; }
[data-testid="stMetricValue"] { font-size: 36px !important; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:.5rem; font-weight:600; }
.badge-green { background:#16a34a22; color:#166534; border:1px solid #16a34a55;}
.badge-yellow{ background:#ca8a0422; color:#854d0e; border:1px solid #ca8a0455;}
.badge-red   { background:#dc262622; color:#7f1d1d; border:1px solid #dc262655;}
.subtitle { margin-top:-.25rem; color:#345; font-size:1.05rem; }
</style>
"""
st.markdown(BRAND_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("Easy Textil Code")
    st.caption("Seu medidor de eficiência")

    file = st.file_uploader("📥 Carregue um CSV", type=["csv"])

    st.divider()
    st.subheader("Acessibilidade")
    base_font = st.slider("Tamanho do texto", 16, 24, 18)
    st.markdown(f"<style>:root {{ --base-font-size:{base_font}px; }}</style>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Alvos (semáforo)")
    target_oee  = st.slider("Alvo OEE",            0.50, 0.95, 0.85, step=0.01)
    target_disp = st.slider("Alvo Disponibilidade",0.60, 0.98, 0.90, step=0.01)
    target_perf = st.slider("Alvo Performance",    0.60, 0.98, 0.92, step=0.01)
    target_qual = st.slider("Alvo Qualidade",      0.80, 1.00, 0.98, step=0.001)

    st.divider()
    st.subheader("Modelo (opcional)")
    if st.button("⬇️ Baixar CSV modelo"):
        exemplo = pd.DataFrame({
            "PRODUTO": ["Tecido A", "Tecido B"],
            "CLIENTE": ["Cliente 1", "Cliente 2"],
            "QUANTIDADE_PRODUTO": [1000, 800],
            "QUANTIDADE_PRODUZIDA": [980, 760],
            "MAQUINAS_NECESSARIAS": ["Urdideira;Tear", "Tear"],
            "DATA_INICIO": ["2025-09-01 08:00", "2025-09-02 08:00"],
            "TEMPO_PLANEJADO_MIN": [600, 480],
            "TEMPO_PARADA_MAQUINA_MIN": [60, 40],
            "QTD_REFUGADA": [15, 20],
            "PARADA_QUEBRA_MIN": [30, 20],
            "PARADA_SETUP_AJUSTE_MIN": [30, 20],
            "MICROPARADAS_MIN": [10, 8],
            "RENDIMENTO_REFUGO_QTD": [5, 4],
            "DEFEITOS_RETRABALHO_QTD": [10, 16],
        })
        st.download_button("Baixar modelo", exemplo.to_csv(index=False).encode("utf-8"),
                           file_name="easy_textil_code_modelo.csv", mime="text/csv")

# colunas esperadas
REQ_BASE = ["PRODUTO","CLIENTE","QUANTIDADE_PRODUTO","MAQUINAS_NECESSARIAS","DATA_INICIO","TEMPO_PARADA_MAQUINA_MIN","QTD_REFUGADA"]
OPT = ["TEMPO_PLANEJADO_MIN","TEMPO_PLANEJADO","QUANTIDADE_PRODUZIDA","DATA_FIM",
       "PARADA_QUEBRA_MIN","PARADA_SETUP_AJUSTE_MIN","MICROPARADAS_MIN","RENDIMENTO_REFUGO_QTD","DEFEITOS_RETRABALHO_QTD"]

def semaforo(v, alvo, label):
    if np.isnan(v): v = 0.0
    if v >= alvo:   cls = "badge badge-green"
    elif v >= alvo*0.85: cls = "badge badge-yellow"
    else:           cls = "badge badge-red"
    return f"<span class='{cls}'>{label}: {v:.1%} (alvo {alvo:.0%})</span>"

def carregar_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    faltando = [c for c in REQ_BASE if c not in df.columns]
    if faltando:
        st.error(f"Colunas obrigatórias ausentes: {', '.join(faltando)}")
        st.stop()
    for c in OPT:
        if c not in df.columns:
            df[c] = 0 if c != "DATA_FIM" else pd.NaT

    if ("TEMPO_PLANEJADO_MIN" not in df.columns) or (pd.to_numeric(df["TEMPO_PLANEJADO_MIN"], errors="coerce").fillna(0) == 0).all():
        if "TEMPO_PLANEJADO" in df.columns:
            df["TEMPO_PLANEJADO_MIN"] = pd.to_numeric(df["TEMPO_PLANEJADO"], errors="coerce").fillna(0)
        else:
            st.error("Informe 'TEMPO_PLANEJADO_MIN' (min) ou 'TEMPO_PLANEJADO' (min) no CSV.")
            st.stop()

    num_cols = ["QUANTIDADE_PRODUTO","QUANTIDADE_PRODUZIDA","TEMPO_PLANEJADO_MIN","TEMPO_PARADA_MAQUINA_MIN",
                "QTD_REFUGADA","PARADA_QUEBRA_MIN","PARADA_SETUP_AJUSTE_MIN","MICROPARADAS_MIN",
                "RENDIMENTO_REFUGO_QTD","DEFEITOS_RETRABALHO_QTD"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["DATA_INICIO"] = pd.to_datetime(df["DATA_INICIO"], errors="coerce")
    if "DATA_FIM" in df.columns:
        df["DATA_FIM"] = pd.to_datetime(df["DATA_FIM"], errors="coerce")

    if "RECURSO" not in df.columns:
        if "MAQUINAS_NECESSARIAS" in df.columns:
            df["RECURSO"] = df["MAQUINAS_NECESSARIAS"].astype(str).str.split("[;,]").str[0].str.strip()
        else:
            df["RECURSO"] = "Recurso"
    return df

def calcular_oee(df: pd.DataFrame) -> pd.DataFrame:
    df["QUANTIDADE_PRODUZIDA"] = pd.to_numeric(df.get("QUANTIDADE_PRODUZIDA", 0), errors="coerce").fillna(0)
    df["PRODUCAO_REAL"] = df["QUANTIDADE_PRODUZIDA"].where(df["QUANTIDADE_PRODUZIDA"] > 0, df["QUANTIDADE_PRODUTO"])

    df["TEMPO_PRODUCAO_REAL_MIN"] = (df["TEMPO_PLANEJADO_MIN"] - df["TEMPO_PARADA_MAQUINA_MIN"]).clip(lower=0)

    denom_disp = df["TEMPO_PLANEJADO_MIN"].replace(0, pd.NA)
    df["Disponibilidade"] = (df["TEMPO_PRODUCAO_REAL_MIN"] / denom_disp).fillna(0).clip(0, 1)

    ciclo_ideal = (df["TEMPO_PLANEJADO_MIN"] / df["QUANTIDADE_PRODUTO"].replace(0, pd.NA)).fillna(0)
    df["TEMPO_CICLO_IDEAL_MIN"] = ciclo_ideal

    denom_perf = df["TEMPO_PRODUCAO_REAL_MIN"].replace(0, pd.NA)
    df["Performance"] = (df["PRODUCAO_REAL"] * df["TEMPO_CICLO_IDEAL_MIN"] / denom_perf).fillna(0).clip(0, 1)

    denom_q = df["PRODUCAO_REAL"].replace(0, pd.NA)
    df["Qualidade"] = ((df["PRODUCAO_REAL"] - df["QTD_REFUGADA"]) / denom_q).fillna(0).clip(0, 1)

    df["OEE"] = (df["Disponibilidade"] * df["Performance"] * df["Qualidade"]).fillna(0)

    if "DATA_FIM" in df.columns:
        mask = df["DATA_FIM"].isna() & df["DATA_INICIO"].notna()
        df.loc[mask, "DATA_FIM"] = df.loc[mask, "DATA_INICIO"] + pd.to_timedelta(df.loc[mask, "TEMPO_PLANEJADO_MIN"], unit="m")

    for c in ["PARADA_QUEBRA_MIN","PARADA_SETUP_AJUSTE_MIN","MICROPARADAS_MIN","RENDIMENTO_REFUGO_QTD","DEFEITOS_RETRABALHO_QTD"]:
        if c not in df.columns: df[c] = 0

    df["PERDA_DISPON_MIN"] = (df["PARADA_QUEBRA_MIN"] + df["PARADA_SETUP_AJUSTE_MIN"]).fillna(0)
    df["PERDA_PERF_MIN"]   = df["MICROPARADAS_MIN"].fillna(0)
    df["PERDA_QUAL_QTD"]   = (df["RENDIMENTO_REFUGO_QTD"] + df["DEFEITOS_RETRABALHO_QTD"]).fillna(0)

    df["DIA"] = df["DATA_INICIO"].dt.date
    return df

st.title("🧵 Easy Textil Code")
st.markdown("<div class='subtitle'>OEE = Disponibilidade × Performance × Qualidade</div>", unsafe_allow_html=True)

if not file:
    st.info("Envie um **CSV** pela barra lateral. Você pode baixar um modelo.")
    st.stop()

df_raw = carregar_csv(file)
df = calcular_oee(df_raw.copy())

st.subheader("🔎 Filtros")
cfa, cfb, cfc = st.columns([1,1,2])
with cfa:
    data_min, data_max = df["DATA_INICIO"].min(), df["DATA_INICIO"].max()
    d_ini, d_fim = st.date_input("Período (DATA_INICIO)",
        value=(data_min.date() if pd.notna(data_min) else datetime.today().date(),
               data_max.date() if pd.notna(data_max) else datetime.today().date()))
with cfb:
    produtos = sorted(df["PRODUTO"].dropna().unique().tolist())
    sel_prod = st.multiselect("Produtos", produtos, default=produtos[: min(5, len(produtos))])
with cfc:
    st.caption("Dica: reduza o período ou escolha produtos específicos.")

mask = (df["DATA_INICIO"].dt.date >= d_ini) & (df["DATA_INICIO"].dt.date <= d_fim)
if sel_prod: mask &= df["PRODUTO"].isin(sel_prod)
df_view = df.loc[mask].copy()
if df_view.empty:
    st.warning("Nenhum registro para o filtro.")
    st.stop()

def badge_row(v, alvo, rot):
    st.markdown(semaforo(v, alvo, rot), unsafe_allow_html=True)

st.subheader("📌 KPIs (média no filtro)")
k1,k2,k3,k4 = st.columns(4)
m_oee = df_view["OEE"].mean()
m_d   = df_view["Disponibilidade"].mean()
m_p   = df_view["Performance"].mean()
m_q   = df_view["Qualidade"].mean()
with k1: st.metric("OEE", f"{m_oee:.1%}");        badge_row(m_oee, target_oee,  "OEE")
with k2: st.metric("Disponibilidade", f"{m_d:.1%}"); badge_row(m_d,   target_disp, "Disp.")
with k3: st.metric("Performance", f"{m_p:.1%}");     badge_row(m_p,   target_perf, "Perf.")
with k4: st.metric("Qualidade", f"{m_q:.1%}");       badge_row(m_q,   target_qual, "Qual.")

st.divider()
st.subheader("📋 Ordens (amostra até 300)")
st.dataframe(df_view.head(300), use_container_width=True)

st.subheader("📊 OEE por Produto")
g_prod = df_view.groupby("PRODUTO", as_index=False)["OEE"].mean().sort_values("OEE", ascending=False)
fig1 = px.bar(g_prod, x="PRODUTO", y="OEE", text="OEE", title="OEE médio por produto")
fig1.update_traces(texttemplate="%{text:.1%}", textposition="outside")
fig1.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📈 Tendência temporal (OEE diário)")
g_dia = df_view.groupby("DIA", as_index=False)[["OEE","Disponibilidade","Performance","Qualidade"]].mean()
fig2 = px.line(g_dia, x="DIA", y=["OEE","Disponibilidade","Performance","Qualidade"], markers=True,
               title="Média diária — OEE e pilares")
fig2.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🧭 Pareto de perdas")
perdas = pd.DataFrame({
    "Categoria": ["Quebra", "Setup/Ajuste", "Microparadas", "Refugo/Defeitos+Retrabalho"],
    "Valor": [
        df_view["PARADA_QUEBRA_MIN"].sum(),
        df_view["PARADA_SETUP_AJUSTE_MIN"].sum(),
        df_view["MICROPARADAS_MIN"].sum(),
        df_view["RENDIMENTO_REFUGO_QTD"].sum() + df_view["DEFEITOS_RETRABALHO_QTD"].sum(),
    ]
}).sort_values("Valor", ascending=False)
fig3 = px.bar(perdas, x="Categoria", y="Valor", text="Valor", title="Pareto — Maiores perdas")
fig3.update_traces(textposition="outside")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📤 Exportar")
cxa, cxb = st.columns(2)
with cxa:
    st.download_button("Baixar CSV", df_view.to_csv(index=False).encode("utf-8"),
                       file_name="easy_textil_code_resultados.csv", mime="text/csv")
with cxb:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wb:
        df_view.to_excel(wb, index=False, sheet_name="Resultados")
        g_prod.to_excel(wb, index=False, sheet_name="OEE_por_Produto")
        g_dia.to_excel(wb, index=False, sheet_name="Tendencia_Diaria")
        perdas.to_excel(wb, index=False, sheet_name="Pareto_Perdas")
    st.download_button("Baixar Excel (múltiplas abas)", out.getvalue(),
                       file_name="easy_textil_code_resultados.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Trate primeiro as 3 maiores perdas do Pareto e compare metas vs real por turno/recursos.")
