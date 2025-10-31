# app.py
# Easy Textil — Carregador/normalizador de CSV
# Correção: usar a **logo do assets** tanto no topo quanto na barra lateral
# e usar a **mesma imagem como ícone da aba (favicon)**.
# (O restante do app permanece exatamente igual.)

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
# Util: localizar logo/favicon dentro de /assets
# ============================================================
def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# Candidatos de nome para facilitar (use o que você já tem no /assets)
LOGO_CANDIDATES = [
    "assets/logo_easy_textil.png",
    "assets/easy_textil_logo.png",
    "assets/logo.png",
    "assets/easy-textil-logo.png",
]

FAVICON_CANDIDATES = [
    "assets/favicon_easy_textil.png",
    "assets/easy_textil_favicon.png",
    "assets/favicon.png",
    "assets/logo_easy_textil.png",  # usa a própria logo, se não houver favicon dedicado
]

LOGO_PATH = _first_existing(LOGO_CANDIDATES)
FAVICON_PATH = _first_existing(FAVICON_CANDIDATES)

HAS_LOGO = LOGO_PATH is not None
HAS_FAVICON = FAVICON_PATH is not None

# ============================================================
# Configuração da página (usa favicon do assets, se existir)
# ============================================================
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon=Image.open(FAVICON_PATH) if HAS_FAVICON else "🧵",
    layout="wide",
)

# ============================================================
# Header com logo do assets (e fallback se não existir)
# ============================================================
def show_header():
    left, right = st.columns([1, 3])
    with left:
        if HAS_LOGO:
            st.image(LOGO_PATH, caption=None, use_container_width=True)
        else:
            # Fallback simples apenas se a logo não estiver presente
            st.markdown(
                """
                <div style="display:flex;align-items:center;gap:.5rem">
                  <div style="font-size:42px;">🧵</div>
                  <div>
                    <div style="font-weight:700;font-size:22px;line-height:1;">Easy Textil</div>
                    <div style="color:#666; font-size:13px; margin-top:-2px;">Seu medidor de eficiência</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    with right:
        st.title("Easy Textil — Seu medidor de eficiência")
        st.caption("OEE = **Disponibilidade × Performance × Qualidade**")

show_header()

# Logo também na barra lateral usando o mesmo arquivo do assets
if HAS_LOGO:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
else:
    st.sidebar.markdown("### 🧵 Easy Textil")
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

# Sinônimos / aliases para auto-detecção
ALIASES: Dict[str, List[str]] = {
    "CLIENTE": ["cliente", "cliente_nome", "comprador", "pedido", "pedido/cliente", "cliente/pedido"],
    "QUANTIDADE_PRODUTO": ["metros", "quantidade", "qtd", "kg prod.", "kg produzido", "kg", "kg plan.", "kg plan", "kg planificado"],
    "MAQUINAS_NECESSARIAS": ["maquinas_necessarias", "maq. nec.", "turnos necessarios", "turnos necessários", "turnos", "maq.", "maquinas"],
    "DATA_INICIO": ["inicio prod.", "inicio", "data inicio", "data_inicio", "inicio_producao", "início prod.", "início produção"],
    "TEMPO_PARADA_MAQUINA_MIN": ["horas maq.", "parada", "parada (min)", "tempo_parada", "tempo_parada_min", "downtime_min", "downtime", "paradas_min"],
    "QTD_REFUGADA": ["qtd_refugada", "refugo", "kg rest.", "kg resto", "kg sucata", "defeituoso", "perda", "perdas"],
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
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    s = s.replace("/", " ").replace("\\", " ").replace("-", " ")
    s = s.replace(".", "").replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_number(x) -> Optional[float]:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s == "**":
        return pd.NA
    s = s.replace(" ", "")
    s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return pd.NA

def _pt_month_to_num(token: str) -> Optional[int]:
    token = token.strip(". ").lower()[:3]
    return PT_MONTHS.get(token)

def parse_date_ptbr(x) -> Optional[pd.Timestamp]:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s == "" or s == "**":
        return pd.NaT
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt
    except Exception:
        pass
    s2 = re.sub(r"([A-Za-z]{3})\.", r"\1", s, flags=re.IGNORECASE)
    m = re.match(r"^\s*(\d{1,2})\s*[-/ ]\s*([A-Za-z]{3,})\s*[-/ ]?\s*(\d{2,4})?\s*$", s2, flags=re.IGNORECASE)
    if m:
        d = int(m.group(1))
        mon = _pt_month_to_num(m.group(2))
        y = m.group(3)
        if mon is not None:
            if y is None:
                y = datetime.now().year
            else:
                y = int(y)
                if y < 100:
                    y += 2000
            try:
                return pd.Timestamp(year=y, month=mon, day=d)
            except Exception:
                return pd.NaT
    try:
        return pd.to_datetime(s2, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def try_read_csv(file) -> pd.DataFrame:
    content = file.read()
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8-sig", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1 or len(df) == 0:
                    return df
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(content), engine="python")

def suggest_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    normalized = {c: norm(c) for c in columns}
    inv = {v: k for k, v in normalized.items()}
    mapping = {k: None for k in REQUIRED}
    for target, names in ALIASES.items():
        found = None
        for cand in names:
            n = norm(cand)
            if n in inv:
                found = inv[n]
                break
            for nc, orig in inv.items():
                if nc.startswith(n) or n in nc:
                    found = orig
                    break
            if found:
                break
        mapping[target] = found
    if mapping["QUANTIDADE_PRODUTO"] is None:
        for pref in ["Kg PROD.", "Kg Prod.", "Kg PROD", "Kg", "METROS", "Quant.", "Quantidade", "QTD"]:
            for c in columns:
                if norm(c) == norm(pref):
                    mapping["QUANTIDADE_PRODUTO"] = c
                    break
            if mapping["QUANTIDADE_PRODUTO"]:
                break
    return mapping

def extract_cliente_from_pedido(series: pd.Series) -> pd.Series:
    def _extract(s):
        if pd.isna(s):
            return pd.NA
        s = str(s).strip()
        m = re.match(r"^\s*\d+\s+(.*)$", s)
        return m.group(1).strip() if m else s
    return series.apply(_extract)

def to_minutes_from_hours(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: (float(v) * 60) if pd.notna(v) else pd.NA)

def coalesce(s: pd.Series, default):
    return s.fillna(default)

# ============================================================
# Barra lateral — upload, modelo e mapeamento
# ============================================================
uploaded = st.sidebar.file_uploader("📥 Carregue um CSV", type=["csv"], help="Limite ~200MB • CSV")

# CSV Modelo coerente com o pipeline
MODEL_DF = pd.DataFrame(
    {
        "CLIENTE": ["Exemplo Indústria"],
        "QUANTIDADE_PRODUTO": [1000],
        "MAQUINAS_NECESSARIAS": [1],
        "DATA_INICIO": ["2025-01-07"],
        "TEMPO_PARADA_MAQUINA_MIN": [0],
        "QTD_REFUGADA": [0],
        # Colunas opcionais comuns (só para referência)
        "PEDIDO": ["98406 Exemplo Indústria"],
        "METROS": [11500],
        "MÁQ.": ["Tear 01"],
        "HORAS MAQ.": [0.0],
        "Kg PROD.": [980],
        "Kg REST.": [20],
    }
)
st.sidebar.download_button(
    "⬇️ Baixar CSV Modelo",
    data=MODEL_DF.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_modelo.csv",
    mime="text/csv",
)

if uploaded is None:
    st.info("Carregue um CSV na barra lateral à esquerda. Dica: você pode **arrastar e soltar** o arquivo.")
    st.stop()

# ============================================================
# Leitura & pré-visualização do CSV bruto
# ============================================================
raw_df = try_read_csv(uploaded)
orig_cols = list(raw_df.columns)

if raw_df.empty:
    st.error("Não consegui ler dados nesse arquivo. Verifique o separador (`,` ou `;`) e o encoding.")
    st.stop()

st.subheader("📄 Pré-visualização do arquivo bruto")
st.dataframe(raw_df.head(20), use_container_width=True)

# Sugerir mapeamento e permitir editar
suggested = suggest_mapping(orig_cols)

st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Mapeamento das colunas obrigatórias")

mapping: Dict[str, Optional[str]] = {}
for target in REQUIRED:
    mapping[target] = st.sidebar.selectbox(
        f"{target}",
        options=[None] + orig_cols,
        index=( [None] + orig_cols ).index(suggested.get(target)) if suggested.get(target) in orig_cols else 0,
        help=(
            "Selecione a coluna do seu CSV que representa **{0}**."
            .format(target)
        ),
        format_func=lambda x: "— selecione —" if x is None else x,
        key=f"map_{target}"
    )

st.sidebar.markdown(
    """
    *Dicas*:
    - **CLIENTE** pode vir dentro de **PEDIDO** (eu extraio).
    - **QUANTIDADE_PRODUTO** pode ser `METROS` ou `Kg PROD.`.
    - **TEMPO_PARADA_MAQUINA_MIN**: se mapear `HORAS MAQ.`, eu converto para **minutos** automaticamente.
    """
)

# ============================================================
# Normalização (números, datas) e construção do dataset "clean"
# ============================================================
df = raw_df.copy()

# Números: converte padrões BR
for c in df.columns:
    if df[c].dtype == object:
        sample = str(df[c].dropna().astype(str).head(10).tolist())
        if re.search(r"(\d+[\.,]\d+)|(\d{1,3}\.\d{3})", sample) or re.search(r"^\d+$", sample):
            df[c] = df[c].apply(parse_number)

# Datas comuns
for c in df.columns:
    if any(tok in norm(c) for tok in ["data", "inicio", "início", "fim", "entrega"]):
        df[c] = df[c].apply(parse_date_ptbr)

clean = pd.DataFrame()

# CLIENTE
if mapping["CLIENTE"]:
    if norm(mapping["CLIENTE"]) == norm("pedido"):
        clean["CLIENTE"] = extract_cliente_from_pedido(df[mapping["CLIENTE"]])
    else:
        clean["CLIENTE"] = df[mapping["CLIENTE"]].astype("string")
else:
    ped_col = next((c for c in orig_cols if norm(c) == "pedido"), None)
    if ped_col:
        clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]).astype("string")
    else:
        clean["CLIENTE"] = pd.Series(pd.NA, index=df.index, dtype="string")

# QUANTIDADE_PRODUTO
if mapping["QUANTIDADE_PRODUTO"]:
    clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
else:
    q_col = None
    for cand in ["METROS", "Kg PROD.", "Kg PROD", "KG PROD.", "KG PROD", "Quantidade", "QTD"]:
        q_col = next((c for c in orig_cols if norm(c) == norm(cand)), None)
        if q_col:
            break
    clean["QUANTIDADE_PRODUTO"] = df[q_col].apply(parse_number) if q_col else pd.NA

# MAQUINAS_NECESSARIAS
if mapping["MAQUINAS_NECESSARIAS"]:
    clean["MAQUINAS_NECESSARIAS"] = coalesce(df[mapping["MAQUINAS_NECESSARIAS"]].apply(parse_number), 1).astype("Int64")
else:
    turnos = next((c for c in orig_cols if "turnos" in norm(c)), None)
    if turnos:
        clean["MAQUINAS_NECESSARIAS"] = coalesce(df[turnos].apply(parse_number), 1).astype("Int64")
    else:
        clean["MAQUINAS_NECESSARIAS"] = pd.Series(1, index=df.index, dtype="Int64")

# DATA_INICIO
if mapping["DATA_INICIO"]:
    clean["DATA_INICIO"] = df[mapping["DATA_INICIO"]].apply(parse_date_ptbr)
else:
    ini_col = next((c for c in orig_cols if "inicio" in norm(c) or "início" in norm(c)), None)
    clean["DATA_INICIO"] = df[ini_col].apply(parse_date_ptbr) if ini_col else pd.NaT

# TEMPO_PARADA_MAQUINA_MIN
if mapping["TEMPO_PARADA_MAQUINA_MIN"]:
    col = mapping["TEMPO_PARADA_MAQUINA_MIN"]
    if "hora" in norm(col):
        clean["TEMPO_PARADA_MAQUINA_MIN"] = to_minutes_from_hours(df[col])
    else:
        vals = df[col].apply(parse_number)
        clean["TEMPO_PARADA_MAQUINA_MIN"] = vals
else:
    horas_col = next((c for c in orig_cols if "hora" in norm(c) and "maq" in norm(c)), None)
    if horas_col:
        clean["TEMPO_PARADA_MAQUINA_MIN"] = to_minutes_from_hours(df[horas_col].apply(parse_number))
    else:
        clean["TEMPO_PARADA_MAQUINA_MIN"] = pd.Series(0, index=df.index, dtype="Float64")

# QTD_REFUGADA
if mapping["QTD_REFUGADA"]:
    clean["QTD_REFUGADA"] = coalesce(df[mapping["QTD_REFUGADA"]].apply(parse_number), 0).astype("Float64")
else:
    rest_col = next((c for c in orig_cols if "kg" in norm(c) and "rest" in norm(c)), None)
    if rest_col:
        tmp = df[rest_col].apply(parse_number)
        tmp = tmp.apply(lambda v: max(v, 0) if pd.notna(v) else v)
        clean["QTD_REFUGADA"] = coalesce(tmp, 0).astype("Float64")
    else:
        clean["QTD_REFUGADA"] = pd.Series(0, index=df.index, dtype="Float64")

# ============================================================
# Validação e feedback (com alerta amigável)
# ============================================================
missing = [c for c in REQUIRED if c not in clean.columns or clean[c].isna().all()]
if missing:
    st.error(
        "⚠️ Colunas obrigatórias ausentes ou vazias: **{}**. "
        "Ajuste o mapeamento na barra lateral.\n\n"
        "Dica: Baixe o **CSV Modelo** para ver exemplos de preenchimento."
        .format(", ".join(missing))
    )
else:
    st.success("✅ Tudo certo! Colunas obrigatórias mapeadas com sucesso.")

# ============================================================
# Visualização do dataset limpo + exportação
# ============================================================
st.subheader("🧼 Dados limpos (prontos para o cálculo de OEE)")
st.dataframe(clean.head(50), use_container_width=True)

out_csv = clean.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV LIMPO (compatível)", data=out_csv, file_name="easy_textil_limpo.csv", mime="text/csv")

# ============================================================
# (Opcional) Métricas rápidas — para dar contexto a leigos
# ============================================================
with st.expander("🧮 (Opcional) Métricas rápidas para conferência"):
    total_prod = pd.to_numeric(clean["QUANTIDADE_PRODUTO"], errors="coerce").sum(min_count=1)
    total_ref = pd.to_numeric(clean["QTD_REFUGADA"], errors="coerce").sum(min_count=1)
    tempo_parada_min = pd.to_numeric(clean["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").sum(min_count=1)
    maquinas_media = pd.to_numeric(clean["MAQUINAS_NECESSARIAS"], errors="coerce").mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quant. Produzida (soma)", f"{total_prod:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(total_prod) else "—")
    col2.metric("Refugo (soma)", f"{total_ref:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(total_ref) else "—")
    col3.metric("Paradas (min, soma)", f"{tempo_parada_min:,.0f}".replace(",", ".") if pd.notna(tempo_parada_min) else "—")
    col4.metric("Máquinas necessárias (média)", f"{maquinas_media:.2f}".replace(".", ",") if pd.notna(maquinas_media) else "—")

st.caption("💡 Quer ligar este carregador direto ao cálculo de OEE (Disponibilidade, Performance, Qualidade) da sua tela principal? Posso integrar no próximo passo.")
