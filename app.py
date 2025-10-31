# app.py
# Easy Textil — Carregador/normalizador de CSV com logo fixa em assets/logo.jpg
# Mantém compatibilidade e adiciona melhorias de UX, mapeamento assistido e insights.

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
    raise FileNotFoundError("❌ Logo não encontrada! Coloque 'logo.jpg' na pasta /assets.")

logo_image = Image.open(LOGO_PATH)

# ============================================================
# Config da página (favicon usa a logo)
# ============================================================
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon=logo_image,
    layout="wide",
)

# ============================================================
# Estilos (look & feel)
# ============================================================
CUSTOM_CSS = """
<style>
/* Tipografia e largura máxima do conteúdo */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
/* Título + subtítulo */
h1 { font-weight: 800; letter-spacing: -0.5px; }
.small-muted { color:#6b7280; font-size:0.92rem; }
/* Chips de status */
.chip { display:inline-flex; align-items:center; gap:.4rem; padding:.20rem .55rem; 
        border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem; }
.chip-ok { background:#DCFCE7; color:#065F46; border:1px solid #86EFAC; }
.chip-warn { background:#FEF9C3; color:#854D0E; border:1px solid #FDE68A; }
.chip-bad { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
/* Cartões (metrics) */
.card { border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; background:white; }
.card h3 { margin:0; font-size:0.95rem; color:#6b7280; font-weight:600; }
.card .value { font-size:1.35rem; font-weight:800; margin-top:.2rem; }
/* Tabelas: linhas zebrada e fonte compacta */
.dataframe td, .dataframe th { font-size: 0.88rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# Header com logo e título
# ============================================================
def show_header():
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image(logo_image, caption=None, use_container_width=True)
    with col_title:
        st.title("Easy Textil — Seu medidor de eficiência")
        st.markdown('<div class="small-muted">OEE = <b>Disponibilidade × Performance × Qualidade</b></div>',
                    unsafe_allow_html=True)

show_header()

# ============================================================
# Sidebar (logo + passo a passo + upload)
# ============================================================
st.sidebar.image(logo_image, use_container_width=True)
st.sidebar.caption("Seu medidor de eficiência")

with st.sidebar.expander("📋 Passo a passo (bem simples) — clique para ver", expanded=True):
    st.markdown(
        """
1) **Baixe o CSV Modelo** e veja os nomes que o sistema espera.  
2) **Arraste seu CSV** ou clique em *Carregar*.  
3) **Confira o mapeamento** das colunas (aba *Mapeamento*).  
4) Veja os **Dados limpos** e baixe o **CSV LIMPO**.  
5) (Opcional) Explore os **Insights**.
        """
    )

# CSV modelo para download
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

uploaded = st.sidebar.file_uploader("📥 Carregue um CSV", type=["csv"])

# ============================================================
# Dicionários/aliases e utilitários
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
    # fontes candidatas no CSV da fábrica
    "CLIENTE": ["cliente", "pedido", "cliente/pedido"],
    "QUANTIDADE_PRODUTO": ["metros", "quantidade", "kg prod.", "kg produzido", "kg"],
    "MAQUINAS_NECESSARIAS": ["turnos necessarios", "turnos", "maq.", "maquinas", "maquinas necessarias"],
    "DATA_INICIO": ["inicio prod.", "inicio producao", "data inicio", "inicio"],
    "TEMPO_PARADA_MAQUINA_MIN": ["horas maq.", "parada", "tempo parada", "parada (h)", "parada (min)"],
    "QTD_REFUGADA": ["kg rest.", "refugo", "kg refugado", "qtd refugado"],
}

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12
}

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
    s = str(x).strip()
    # remove milhar e troca vírgula por ponto
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA

def parse_date_ptbr(x) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s in ("", "**"):
        return pd.NaT
    # tenta com pandas primeiro
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return dt
    # tenta formas como "12-dez." "7-jan." "09/jan" "23-jan-24"
    m = re.match(r"(\d{1,2})[-/ ]([A-Za-z]{3,})\.?[-/ ]?(\d{2,4})?$", s)
    if m:
        d = int(m.group(1))
        mon = PT_MONTHS.get(m.group(2)[:3].lower())
        ytxt = m.group(3)
        if ytxt is None or ytxt == "":
            y = datetime.now().year
        elif len(ytxt) == 2:
            y = 2000 + int(ytxt)
        else:
            y = int(ytxt)
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

@st.cache_data(show_spinner=False)
def try_read_csv(content: bytes) -> pd.DataFrame:
    # tenta separadores/encodes
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc)
                # heurística: >1 coluna
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    # fallback
    return pd.read_csv(io.BytesIO(content))

# ============================================================
# Sessão (mapeamento manual persistente)
# ============================================================
if "mapping_manual" not in st.session_state:
    st.session_state.mapping_manual = {}

# ============================================================
# Early exit se não tem arquivo
# ============================================================
if uploaded is None:
    st.info("👈 Carregue um arquivo CSV na barra lateral para começar.")
    st.stop()

# ============================================================
# Leitura do CSV e pré-visualização
# ============================================================
content = uploaded.read()
df_raw = try_read_csv(content)
df = df_raw.copy()

st.markdown("### 📄 Pré-visualização do CSV")
st.dataframe(df.head(15), use_container_width=True)

# ============================================================
# Sugestão de mapeamento automático
# ============================================================
cols = list(df.columns)
auto_mapping: Dict[str, Optional[str]] = {}
for key, aliases in ALIASES.items():
    found = None
    for col in cols:
        if norm(col) in [norm(a) for a in aliases]:
            found = col
            break
    auto_mapping[key] = found

# ============================================================
# Aba de Mapeamento (assistido)
# ============================================================
tab_prev, tab_map, tab_clean, tab_insights, tab_dl = st.tabs(
    ["🔎 Pré-visualização", "🧭 Mapeamento", "🧼 Dados limpos", "📈 Insights", "⬇️ Downloads"]
)

with tab_map:
    st.subheader("🧭 Mapeamento das colunas (confirme/ajuste)")
    st.caption("O sistema sugere automaticamente; se precisar, ajuste manualmente e clique em **Aplicar mapeamento**.")

    # desenha selects lado a lado
    sel_cols = st.columns(3)
    sel_cols2 = st.columns(3)
    pickers = {}
    keys_sorted = REQUIRED.copy()

    for i, key in enumerate(keys_sorted):
        target_col = cols.copy()
        target_col.insert(0, "— (não usar) —")
        default = auto_mapping.get(key) if auto_mapping.get(key) in target_col else "— (não usar) —"
        container = sel_cols[i] if i < 3 else sel_cols2[i-3]
        with container:
            pickers[key] = st.selectbox(
                f"{key}",
                options=target_col,
                index=target_col.index(default) if default in target_col else 0,
                key=f"picker_{key}",
            )

    if st.button("✅ Aplicar mapeamento", type="primary"):
        st.session_state.mapping_manual = {
            k: (None if v == "— (não usar) —" else v) for k, v in pickers.items()
        }
        st.success("Mapeamento salvo para esta sessão.")

    # Exibe status de cada campo
    st.markdown("#### Status dos requisitos")
    final_map_preview = {k: st.session_state.mapping_manual.get(k) or auto_mapping.get(k) for k in REQUIRED}
    chips = []
    for k in REQUIRED:
        if final_map_preview.get(k):
            chips.append(f'<span class="chip chip-ok">✓ {k}</span>')
        else:
            chips.append(f'<span class="chip chip-bad">• {k} faltando</span>')
    st.markdown(" ".join(chips), unsafe_allow_html=True)

# ============================================================
# Função que normaliza dados segundo o mapeamento final
# ============================================================
def build_clean(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    clean = pd.DataFrame(index=df.index)

    # CLIENTE
    if mapping.get("CLIENTE"):
        clean["CLIENTE"] = df[mapping["CLIENTE"]].astype(str)
    else:
        ped_col = next((c for c in df.columns if "pedido" in norm(c)), None)
        clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]) if ped_col else pd.NA

    # QUANTIDADE_PRODUTO
    if mapping.get("QUANTIDADE_PRODUTO"):
        clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
    else:
        qtd_col = next((c for c in df.columns if "metro" in norm(c)), None)
        clean["QUANTIDADE_PRODUTO"] = df[qtd_col].apply(parse_number) if qtd_col else pd.NA

    # MAQUINAS_NECESSARIAS
    if mapping.get("MAQUINAS_NECESSARIAS"):
        clean["MAQUINAS_NECESSARIAS"] = df[mapping["MAQUINAS_NECESSARIAS"]].apply(parse_number).fillna(1).astype("Int64")
    else:
        # fallback por TURNOS NECESSÁRIOS
        t_col = next((c for c in df.columns if "turno" in norm(c)), None)
        if t_col:
            clean["MAQUINAS_NECESSARIAS"] = df[t_col].apply(parse_number).fillna(1).astype("Int64")
        else:
            clean["MAQUINAS_NECESSARIAS"] = 1

    # DATA_INICIO
    if mapping.get("DATA_INICIO"):
        clean["DATA_INICIO"] = df[mapping["DATA_INICIO"]].apply(parse_date_ptbr)
    else:
        # tenta colunas similares (INICIO PROD., INICIO PRODUCAO, etc.)
        candidate = next((c for c in df.columns if "inicio" in norm(c)), None)
        clean["DATA_INICIO"] = df[candidate].apply(parse_date_ptbr) if candidate else pd.NaT

    # TEMPO_PARADA_MAQUINA_MIN
    if mapping.get("TEMPO_PARADA_MAQUINA_MIN"):
        # detecta se veio em horas/minutos pelo nome
        colname = mapping["TEMPO_PARADA_MAQUINA_MIN"]
        vals = df[colname]
        minutes = vals.apply(parse_number)
        if "hora" in norm(colname):  # HORAS MAQ.
            minutes = minutes.apply(lambda h: pd.NA if pd.isna(h) else h * 60)
        clean["TEMPO_PARADA_MAQUINA_MIN"] = minutes.fillna(0)
    else:
        # fallback: se existir HORAS MAQ., converte
        h_col = next((c for c in df.columns if "hora" in norm(c) and "maq" in norm(c)), None)
        if h_col:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = df[h_col].apply(parse_number).apply(
                lambda h: pd.NA if pd.isna(h) else h * 60
            ).fillna(0)
        else:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = 0

    # QTD_REFUGADA
    if mapping.get("QTD_REFUGADA"):
        clean["QTD_REFUGADA"] = df[mapping["QTD_REFUGADA"]].apply(parse_number).fillna(0)
    else:
        r_col = next((c for c in df.columns if "rest" in norm(c) or "refug" in norm(c)), None)
        clean["QTD_REFUGADA"] = df[r_col].apply(parse_number).fillna(0) if r_col else 0

    return clean

# mapeamento final (manual tem prioridade)
final_mapping = {k: st.session_state.mapping_manual.get(k) or auto_mapping.get(k) for k in REQUIRED}

# ============================================================
# Construção dos dados limpos + estilo
# ============================================================
clean_df = build_clean(df, final_mapping)

def style_clean(df_: pd.DataFrame):
    def neg_red(val):
        try:
            return "color:#991B1B;font-weight:600" if float(val) < 0 else ""
        except Exception:
            return ""
    def missing_yellow(val):
        return "background-color:#FEF9C3" if (pd.isna(val) or val == "" or str(val) == "NaT") else ""
    styled = (
        df_.style
        .applymap(neg_red, subset=["QTD_REFUGADA", "TEMPO_PARADA_MAQUINA_MIN"])
        .applymap(missing_yellow)
        .format(precision=2, na_rep="—")
    )
    return styled

with tab_clean:
    st.subheader("🧼 Dados limpos")
    # cartões rápidos
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="card"><h3>Linhas</h3><div class="value">{:,}</div></div>'.format(len(clean_df)), unsafe_allow_html=True)
    with c2:
        ok_cols = sum((clean_df[c].notna().sum() if c in clean_df.columns else 0) for c in ["CLIENTE"])
        st.markdown('<div class="card"><h3>Clientes (preenchidos)</h3><div class="value">{:,}</div></div>'.format(ok_cols), unsafe_allow_html=True)
    with c3:
        media_parada = float(pd.to_numeric(clean_df["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").fillna(0).mean())
        st.markdown('<div class="card"><h3>Média parada (min)</h3><div class="value">{:.1f}</div></div>'.format(media_parada), unsafe_allow_html=True)
    with c4:
        ref_total = float(pd.to_numeric(clean_df["QTD_REFUGADA"], errors="coerce").fillna(0).sum())
        st.markdown('<div class="card"><h3>Refugo total</h3><div class="value">{:.1f}</div></div>'.format(ref_total), unsafe_allow_html=True)

    st.dataframe(style_clean(clean_df).to_html(), use_container_width=True, unsafe_allow_html=True)

with tab_insights:
    st.subheader("📈 Insights rápidos")
    colA, colB = st.columns([2,1])
    with colA:
        # pequeno gráfico por DATA_INICIO (quantidade por dia)
        df_plot = clean_df.copy()
        df_plot["DATA_INICIO"] = pd.to_datetime(df_plot["DATA_INICIO"], errors="coerce")
        grp = (
            df_plot
            .dropna(subset=["DATA_INICIO"])
            .groupby(pd.Grouper(key="DATA_INICIO", freq="D"))["QUANTIDADE_PRODUTO"]
            .sum()
            .reset_index()
        )
        if not grp.empty:
            st.line_chart(grp.set_index("DATA_INICIO")["QUANTIDADE_PRODUTO"])
        else:
            st.info("Sem datas válidas suficientes para gráfico.")
    with colB:
        # checklist de requisitos
        st.markdown("#### Requisitos")
        req_ok = []
        for k in REQUIRED:
            if k in clean_df.columns and clean_df[k].notna().any():
                req_ok.append(f'<span class="chip chip-ok">✓ {k}</span>')
            else:
                req_ok.append(f'<span class="chip chip-warn">• {k} vazio ou ausente</span>')
        st.markdown(" ".join(req_ok), unsafe_allow_html=True)
        st.caption("Dica: ajuste o mapeamento na aba **Mapeamento**.")

with tab_dl:
    st.subheader("⬇️ Exportar")
    st.download_button(
        "Baixar CSV LIMPO",
        data=clean_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="easy_textil_limpo.csv",
        type="primary",
    )
    st.caption("O arquivo exportado contém somente as colunas normalizadas requeridas pelo pipeline.")

# Mantém a primeira aba como “espelho” da pré-visualização original
with tab_prev:
    st.info("Esta aba mostra as primeiras linhas do arquivo exatamente como foi lido (sem transformações).")
    st.dataframe(df.head(30), use_container_width=True)
