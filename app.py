# app.py
# Easy Textil v2 — CSV normalizador com OEE, mapeamento assistido, dashboard, lote .zip,
# cache de mapeamentos por arquivo e integração opcional com Google Sheets.
# Logo: assets/logo.jpg (obrigatória) | Hero: assets/hero.jpg (opcional)
# Dedicatória: by Reistec Tecnologia

import io
import re
import os
import json
import zipfile
import hashlib
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

# (opcional) Plotly para gráficos mais bonitos. Se não houver, usa fallback.
try:
    import plotly.express as px
except Exception:
    px = None

# (opcional) Google Sheets via gspread
try:
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
except Exception:
    gspread = None

# ============================================================
# ------------------ ASSETS (logo / hero) --------------------
# ============================================================
LOGO_PATH = "assets/logo.jpg"
HERO_PATH = "assets/hero.jpg"

if not os.path.exists(LOGO_PATH):
    raise FileNotFoundError("❌ Coloque sua logo em 'assets/logo.jpg'.")

logo_image = Image.open(LOGO_PATH)
hero_image = Image.open(HERO_PATH) if os.path.exists(HERO_PATH) else None

# ============================================================
# -------------------- CONFIG DA PÁGINA ----------------------
# ============================================================
st.set_page_config(
    page_title="Easy Textil — Seu medidor de eficiência",
    page_icon=logo_image,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# -------------------------- CSS -----------------------------
# ============================================================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; max-width: 1400px;}
      h1 { font-weight: 800; letter-spacing: -0.5px; }
      h3 { font-weight: 700; }
      .small-muted { color:#6b7280; font-size:0.92rem; }
      .hero {
        border-radius:16px; padding:22px;
        background:linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        margin-bottom: 1rem;
        border:1px solid #dbeafe;
      }
      .hero h1 { margin:0; color:#1E3A8A; font-weight:800; }
      .hero .sub { color:#334155; font-size:1rem; }
      .card {
        border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px;
        background:white; box-shadow:0 1px 2px rgba(0,0,0,0.04);
        text-align:center;
      }
      .card h3 { margin:0; font-size:0.95rem; color:#6b7280; font-weight:600; }
      .card .value { font-size:1.4rem; font-weight:800; margin-top:.25rem; color:#111827; }
      .chip {
        display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .6rem;
        border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem;
        border:1px solid transparent;
      }
      .chip-ok  { background:#DCFCE7; color:#065F46; border-color:#86EFAC; }
      .chip-bad { background:#FEE2E2; color:#991B1B; border-color:#FCA5A5; }
      .chip-warn{ background:#FEF9C3; color:#854D0E; border-color:#FDE68A; }
      .muted {color:#6b7280;}
      .stMarkdown a { text-decoration: none; }
      div[data-baseweb="select"] { min-height: 38px; }
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ---------------- SIDEBAR / PASSO A PASSO -------------------
# ============================================================
st.sidebar.image(logo_image, use_container_width=True)
st.sidebar.caption("Seu medidor de eficiência")

with st.sidebar.expander("📋 Passo a passo (bem simples)", expanded=True):
    st.markdown(
        """
1) **Baixe o CSV Modelo**.  
2) **Envie um CSV** (ou um **.zip** com vários CSVs).  
3) Ajuste o **Mapeamento** (se necessário).  
4) Veja **Dados Limpos**, **OEE** e **Dashboard**.  
5) **Exporte** ou **envie ao Google Sheets**.
        """
    )

# CSV modelo
MODEL_DF = pd.DataFrame({
    "CLIENTE": ["Exemplo Indústria"],
    "QUANTIDADE_PRODUTO": [1000],
    "MAQUINAS_NECESSARIAS": [1],
    "DATA_INICIO": ["2025-01-07"],
    "TEMPO_PARADA_MAQUINA_MIN": [0],
    "QTD_REFUGADA": [0],
    # auxiliares úteis p/ OEE e dashboard
    "TURNOS_NECESSARIOS": [1],
    "QUANTIDADE_PLANEJADA": [1000],
    "HORAS_MAQ": [0],
    # campos que podem existir nos seus CSVs
    "MÁQ.": ["T01"],
})
st.sidebar.download_button(
    "⬇️ Baixar CSV Modelo",
    data=MODEL_DF.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_modelo.csv",
)

uploaded_csv = st.sidebar.file_uploader("📥 CSV único", type=["csv"], key="file_csv")
uploaded_zip = st.sidebar.file_uploader("📦 Lote (.zip)", type=["zip"], key="file_zip")

# ============================================================
# ----------------- COLUNAS / ALIASES / REGRAS ---------------
# ============================================================
REQUIRED = [
    "CLIENTE",
    "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS",
    "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN",
    "QTD_REFUGADA",
]
OPTIONAL = [
    "TURNOS_NECESSARIOS",
    "QUANTIDADE_PLANEJADA",
    "KG_PLANEJADO",
    "KG_PRODUZIDO",
    "HORAS_MAQ",
    # para dashboard (se existir em seus arquivos)
    "MÁQ.", "MAQ", "MAQUINA", "MAQUINAS", "MAQUINA_ID"
]

ALIASES: Dict[str, List[str]] = {
    "CLIENTE": ["cliente", "pedido", "cliente/pedido", "cliente - pedido"],
    "QUANTIDADE_PRODUTO": ["metros", "quantidade", "kg prod.", "kg produzido", "kg", "qtd", "kg produzidos", "quantidade produzida"],
    "MAQUINAS_NECESSARIAS": ["maquinas necessarias", "maq.", "maquinas", "turnos", "turnos necessarios"],
    "DATA_INICIO": ["inicio prod.", "inicio producao", "data inicio", "inicio", "inicio prod", "data de inicio"],
    "TEMPO_PARADA_MAQUINA_MIN": ["parada (min)", "parada", "tempo parada", "minutos parada", "tempo parado"],
    "QTD_REFUGADA": ["kg rest.", "refugo", "kg refugado", "qtd refugado", "kg rest", "qtd resto"],
    # opcionais
    "TURNOS_NECESSARIOS": ["turnos necessarios", "turnos"],
    "QUANTIDADE_PLANEJADA": ["quantidade planejada", "planejado", "kg. plan.", "kg plan", "kg planejado", "qtd planejada"],
    "KG_PLANEJADO": ["kg. plan.", "kg planejado", "planejado kg"],
    "KG_PRODUZIDO": ["kg prod.", "kg produzido", "kg produzidos"],
    "HORAS_MAQ": ["horas maq.", "horas maquina", "h maquina", "hora maquina"],
}

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12
}

# ============================================================
# --------------------- FUNÇÕES ÚTEIS ------------------------
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
    s = str(x).strip()
    if s == "":
        return pd.NA
    s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA

def parse_date_ptbr(x) -> Optional[pd.Timestamp]:
    if pd.isna(x) or str(x).strip() in ("", "**"):
        return pd.NaT
    s = str(x).strip()
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return dt
    m = re.match(r"(\d{1,2})[-/ ]([A-Za-z]{3,})\.?[-/ ]?(\d{2,4})?$", s)
    if m:
        d = int(m.group(1))
        mon = PT_MONTHS.get(m.group(2)[:3].lower())
        ytxt = m.group(3)
        if not ytxt:
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
def try_read_csv_bytes(content: bytes) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(content))

def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    mapping: Dict[str, Optional[str]] = {}
    for key, aliases in ALIASES.items():
        found = None
        for col in cols:
            if norm(col) in [norm(a) for a in aliases]:
                found = col
                break
        mapping[key] = found
    return mapping

def ensure_required_columns(clean: pd.DataFrame) -> pd.DataFrame:
    if "CLIENTE" not in clean:
        clean["CLIENTE"] = pd.Series(["—"] * len(clean), index=clean.index, dtype="object")
    if "QUANTIDADE_PRODUTO" not in clean:
        clean["QUANTIDADE_PRODUTO"] = pd.Series([0.0] * len(clean), index=clean.index)
    if "MAQUINAS_NECESSARIAS" not in clean:
        clean["MAQUINAS_NECESSARIAS"] = pd.Series([1] * len(clean), index=clean.index, dtype="Int64")
    if "DATA_INICIO" not in clean:
        clean["DATA_INICIO"] = pd.Series([pd.NaT] * len(clean), index=clean.index, dtype="datetime64[ns]")
    if "TEMPO_PARADA_MAQUINA_MIN" not in clean:
        clean["TEMPO_PARADA_MAQUINA_MIN"] = pd.Series([0.0] * len(clean), index=clean.index)
    if "QTD_REFUGADA" not in clean:
        clean["QTD_REFUGADA"] = pd.Series([0.0] * len(clean), index=clean.index)
    if "TURNOS_NECESSARIOS" not in clean:
        clean["TURNOS_NECESSARIOS"] = pd.Series([pd.NA] * len(clean), index=clean.index)
    if "QUANTIDADE_PLANEJADA" not in clean:
        clean["QUANTIDADE_PLANEJADA"] = pd.Series([pd.NA] * len(clean), index=clean.index)
    if "HORAS_MAQ" not in clean:
        clean["HORAS_MAQ"] = pd.Series([pd.NA] * len(clean), index=clean.index)
    return clean

def build_clean(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    clean = pd.DataFrame(index=df.index)

    # CLIENTE
    if mapping.get("CLIENTE"):
        clean["CLIENTE"] = df[mapping["CLIENTE"]].astype(str)
    else:
        ped_col = next((c for c in df.columns if "pedido" in norm(c)), None)
        clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]) if ped_col else pd.Series(["—"] * len(df))

    # QUANTIDADE_PRODUTO
    if mapping.get("KG_PRODUZIDO"):
        clean["QUANTIDADE_PRODUTO"] = df[mapping["KG_PRODUZIDO"]].apply(parse_number)
    elif mapping.get("QUANTIDADE_PRODUTO"):
        clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
    else:
        qtd_col = next((c for c in df.columns if any(w in norm(c) for w in ["metro", "quantidade", "kg"])), None)
        clean["QUANTIDADE_PRODUTO"] = df[qtd_col].apply(parse_number) if qtd_col else pd.Series([0.0] * len(df))

    # MAQUINAS_NECESSARIAS (ou TURNOS)
    if mapping.get("MAQUINAS_NECESSARIAS"):
        clean["MAQUINAS_NECESSARIAS"] = df[mapping["MAQUINAS_NECESSARIAS"]].apply(parse_number).fillna(1).astype("Int64")
    else:
        t_col = mapping.get("TURNOS_NECESSARIOS") or next((c for c in df.columns if "turno" in norm(c)), None)
        clean["MAQUINAS_NECESSARIAS"] = df[t_col].apply(parse_number).fillna(1).astype("Int64") if t_col else pd.Series([1] * len(df), dtype="Int64")

    # DATA_INICIO
    if mapping.get("DATA_INICIO"):
        clean["DATA_INICIO"] = df[mapping["DATA_INICIO"]].apply(parse_date_ptbr)
    else:
        candidate = next((c for c in df.columns if "inicio" in norm(c) or "data" in norm(c)), None)
        clean["DATA_INICIO"] = df[candidate].apply(parse_date_ptbr) if candidate else pd.Series([pd.NaT] * len(df))

    # TEMPO_PARADA_MAQUINA_MIN
    if mapping.get("TEMPO_PARADA_MAQUINA_MIN"):
        minutes = df[mapping["TEMPO_PARADA_MAQUINA_MIN"]].apply(parse_number)
        clean["TEMPO_PARADA_MAQUINA_MIN"] = minutes.fillna(0)
    else:
        h_col = mapping.get("HORAS_MAQ") or next((c for c in df.columns if "hora" in norm(c) and "maq" in norm(c)), None)
        if h_col:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = df[h_col].apply(parse_number).apply(lambda h: pd.NA if pd.isna(h) else h*60).fillna(0)
        else:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = pd.Series([0.0] * len(df))

    # QTD_REFUGADA
    if mapping.get("QTD_REFUGADA"):
        clean["QTD_REFUGADA"] = df[mapping["QTD_REFUGADA"]].apply(parse_number).fillna(0)
    else:
        r_col = next((c for c in df.columns if "rest" in norm(c) or "refug" in norm(c)), None)
        clean["QTD_REFUGADA"] = df[r_col].apply(parse_number).fillna(0) if r_col else pd.Series([0.0] * len(df))

    # auxiliares
    if mapping.get("TURNOS_NECESSARIOS"):
        clean["TURNOS_NECESSARIOS"] = df[mapping["TURNOS_NECESSARIOS"]].apply(parse_number).fillna(pd.NA)
    else:
        clean["TURNOS_NECESSARIOS"] = pd.Series([pd.NA] * len(df))
    if mapping.get("QUANTIDADE_PLANEJADA"):
        clean["QUANTIDADE_PLANEJADA"] = df[mapping["QUANTIDADE_PLANEJADA"]].apply(parse_number)
    elif mapping.get("KG_PLANEJADO"):
        clean["QUANTIDADE_PLANEJADA"] = df[mapping["KG_PLANEJADO"]].apply(parse_number)
    else:
        clean["QUANTIDADE_PLANEJADA"] = pd.Series([pd.NA] * len(df))
    if mapping.get("HORAS_MAQ"):
        clean["HORAS_MAQ"] = df[mapping["HORAS_MAQ"]].apply(parse_number)
    else:
        clean["HORAS_MAQ"] = pd.Series([pd.NA] * len(df))

    clean = ensure_required_columns(clean)
    return clean

def compute_oee(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    tp_turnos = pd.to_numeric(df.get("TURNOS_NECESSARIOS"), errors="coerce") * (8 * 60)
    tp_maqs   = pd.to_numeric(df.get("MAQUINAS_NECESSARIAS"), errors="coerce") * (8 * 60)
    tempo_planejado = tp_turnos.fillna(tp_maqs).fillna(8 * 60)

    paradas = pd.to_numeric(df["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").fillna(0)
    tempo_operacao = (tempo_planejado - paradas).clip(lower=0)

    disponibilidade = (tempo_operacao / tempo_planejado).fillna(0).clip(0, 1)

    produzido = pd.to_numeric(df["QUANTIDADE_PRODUTO"], errors="coerce").fillna(0)
    planejado = pd.to_numeric(df.get("QUANTIDADE_PLANEJADA"), errors="coerce")

    performance = pd.Series(1.0, index=df.index)
    mask_perf = planejado.notna() & (planejado > 0)
    performance.loc[mask_perf] = (produzido.loc[mask_perf] / planejado.loc[mask_perf]).clip(lower=0)
    performance = performance.clip(upper=2.0)

    refugado = pd.to_numeric(df["QTD_REFUGADA"], errors="coerce").fillna(0)
    qualidade = pd.Series(1.0, index=df.index)
    mask_q = (produzido > 0)
    qualidade.loc[mask_q] = ((produzido.loc[mask_q] - refugado.loc[mask_q]).clip(lower=0) / produzido.loc[mask_q])
    qualidade = qualidade.fillna(1).clip(0, 1)

    oee = (disponibilidade * performance * qualidade).clip(0)

    out = df.copy()
    out["TEMPO_PLANEJADO_MIN"] = tempo_planejado
    out["TEMPO_OPERACAO_MIN"]  = tempo_operacao
    out["DISPONIBILIDADE"]     = disponibilidade
    out["PERFORMANCE"]         = performance
    out["QUALIDADE"]           = qualidade
    out["OEE"]                 = oee
    return out

def mapping_status(mapping: Dict[str, Optional[str]]) -> str:
    chips = []
    for k in REQUIRED:
        chips.append(
            f'<span class="chip {"chip-ok" if mapping.get(k) else "chip-bad"}">'
            f'{"✓ " if mapping.get(k) else "• "}{k}</span>'
        )
    return " ".join(chips)

# ============================================================
# ------------------ CACHE DE MAPEAMENTOS --------------------
# ============================================================
def file_fingerprint(name: str, content: bytes) -> str:
    """Cria uma chave única por arquivo (nome + hash dos bytes)."""
    h = hashlib.sha256(content).hexdigest()[:16]
    return f"{name}::{h}"

if "map_cache" not in st.session_state:
    st.session_state.map_cache: Dict[str, Dict[str, Optional[str]]] = {}

# ============================================================
# -------------------------- HERO ----------------------------
# ============================================================
if hero_image:
    st.image(hero_image, use_container_width=True)
else:
    st.markdown(
        """
        <div class="hero">
          <h1>Easy Textil — Seu medidor de eficiência</h1>
          <div class="sub">OEE = <b>Disponibilidade × Performance × Qualidade</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# --------------------------- TABS ---------------------------
# ============================================================
tab_prev, tab_map, tab_clean, tab_oee, tab_dash, tab_batch, tab_sheets, tab_dl = st.tabs(
    ["🔎 Pré-visualização", "🧭 Mapeamento", "🧼 Dados limpos", "📊 OEE", "📈 Dashboard", "📦 Lote (.zip)", "🟢 Google Sheets", "⬇️ Downloads"]
)

# ================== ENTRADA CSV ÚNICO ======================
df_raw = None
csv_key = None
if uploaded_csv is not None:
    csv_bytes = uploaded_csv.read()
    csv_key = file_fingerprint(uploaded_csv.name, csv_bytes)
    df_raw = try_read_csv_bytes(csv_bytes)

# ================== PRÉ-VISUALIZAÇÃO =======================
with tab_prev:
    if df_raw is not None:
        st.markdown("### 📄 Pré-visualização do CSV")
        st.dataframe(df_raw.head(30), use_container_width=True)
    else:
        st.info("👈 Envie um CSV para visualizar aqui.")

# ======================= MAPEAMENTO =========================
with tab_map:
    if df_raw is None:
        st.info("Carregue um CSV para configurar o mapeamento.")
    else:
        auto_map = auto_map_columns(df_raw)

        # tenta recuperar cache
        cached_map = st.session_state.map_cache.get(csv_key, {}) if csv_key else {}

        st.subheader("🧭 Confirme/ajuste o mapeamento")
        st.caption("Sugiro automaticamente. Você pode ajustar e salvar no cache por arquivo.")

        cols = list(df_raw.columns)
        grid = st.columns(3)
        pickers: Dict[str, Optional[str]] = {}
        keys = REQUIRED + [k for k in OPTIONAL if k not in REQUIRED]

        for i, key in enumerate(keys):
            options = ["— (não usar) —"] + cols
            default = (
                cached_map.get(key) or
                st.session_state.get(f"map_selected_{key}") or
                auto_map.get(key)
            )
            default = default if (default in cols) else "— (não usar) —"
            value = grid[i % 3].selectbox(
                key,
                options=options,
                index=options.index(default) if default in options else 0,
                key=f"map_select_{key}",
            )
            pickers[key] = None if value == "— (não usar) —" else value
            st.session_state[f"map_selected_{key}"] = pickers[key]

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ Aplicar (sessão)", type="primary"):
                st.success("Mapeamento aplicado nesta sessão.")
        with c2:
            if st.button("💾 Salvar no cache deste arquivo"):
                if csv_key:
                    st.session_state.map_cache[csv_key] = pickers.copy()
                    st.success("Mapeamento salvo no cache para este arquivo.")
                else:
                    st.warning("Envie um CSV primeiro.")
        with c3:
            if st.button("🧹 Limpar cache deste arquivo"):
                if csv_key and csv_key in st.session_state.map_cache:
                    st.session_state.map_cache.pop(csv_key, None)
                    st.success("Cache removido para este arquivo.")
                else:
                    st.info("Nenhum cache salvo ainda para este arquivo.")

        final_map = {k: pickers.get(k) or auto_map.get(k) for k in (REQUIRED + OPTIONAL)}
        st.markdown("#### Status")
        st.markdown(mapping_status(final_map), unsafe_allow_html=True)

# ====================== DADOS LIMPOS ========================
clean_df = None
if df_raw is not None:
    # final map = cache (se existir) > seleção sessão > automap
    base_map = st.session_state.map_cache.get(csv_key, {}) if csv_key else {}
    final_map = {k: base_map.get(k) or st.session_state.get(f"map_selected_{k}") or auto_map_columns(df_raw).get(k)
                 for k in (REQUIRED + OPTIONAL)}
    clean_df = build_clean(df_raw, final_map)

with tab_clean:
    if clean_df is None:
        st.info("Carregue um CSV e aplique o mapeamento para ver os dados limpos.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="card"><h3>Linhas</h3><div class="value">{len(clean_df):,}</div></div>', unsafe_allow_html=True)
        with c2:
            ref_total = float(pd.to_numeric(clean_df.get("QTD_REFUGADA"), errors="coerce").fillna(0).sum())
            st.markdown(f'<div class="card"><h3>Refugo total</h3><div class="value">{ref_total:.1f}</div></div>', unsafe_allow_html=True)
        with c3:
            media_parada = float(pd.to_numeric(clean_df.get("TEMPO_PARADA_MAQUINA_MIN"), errors="coerce").fillna(0).mean())
            st.markdown(f'<div class="card"><h3>Média parada (min)</h3><div class="value">{media_parada:.1f}</div></div>', unsafe_allow_html=True)
        with c4:
            clientes_ok = int(clean_df.get("CLIENTE").notna().sum()) if "CLIENTE" in clean_df else 0
            st.markdown(f'<div class="card"><h3>Clientes (preenchidos)</h3><div class="value">{clientes_ok:,}</div></div>', unsafe_allow_html=True)

        st.markdown("#### Tabela (dados limpos)")
        st.dataframe(clean_df.head(200), use_container_width=True)

# ============================ OEE ===========================
oee_df = None
with tab_oee:
    if clean_df is None:
        st.info("Gere os **Dados limpos** primeiro para calcular o OEE.")
    else:
        oee_df = compute_oee(clean_df)

        d_mean = float(oee_df["DISPONIBILIDADE"].mean(skipna=True))
        p_mean = float(oee_df["PERFORMANCE"].mean(skipna=True))
        q_mean = float(oee_df["QUALIDADE"].mean(skipna=True))
        o_mean = float(oee_df["OEE"].mean(skipna=True))

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="card"><h3>Disponibilidade</h3><div class="value">{d_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><h3>Performance</h3><div class="value">{p_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><h3>Qualidade</h3><div class="value">{q_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="card"><h3>OEE Médio</h3><div class="value">{o_mean:0.2%}</div></div>', unsafe_allow_html=True)

        plot_df = oee_df.copy()
        plot_df["DATA_INICIO"] = pd.to_datetime(plot_df["DATA_INICIO"], errors="coerce")
        serie = plot_df.dropna(subset=["DATA_INICIO"]).groupby(pd.Grouper(key="DATA_INICIO", freq="D"))["OEE"].mean()
        if serie.dropna().empty:
            st.info("Sem datas válidas suficientes para o gráfico.")
        else:
            st.line_chart(serie, height=260)

        st.markdown("#### Tabela OEE")
        st.dataframe(oee_df, use_container_width=True)

# ========================= DASHBOARD ========================
with tab_dash:
    if oee_df is None:
        st.info("Gere o **OEE** para ver o dashboard.")
    else:
        st.subheader("📈 Visão geral")

        # ------ TOP CLIENTES por Produção ------
        top_clientes = (
            oee_df.groupby("CLIENTE", dropna=False)["QUANTIDADE_PRODUTO"]
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 clientes por produção**")
            if px:
                fig = px.bar(top_clientes, x="CLIENTE", y="QUANTIDADE_PRODUTO")
                fig.update_layout(height=320, xaxis_title=None, yaxis_title="Produzido")
                st.plotly_chart(fig, use_container_width=True)
            else:
                top_clientes = top_clientes.set_index("CLIENTE")
                st.bar_chart(top_clientes, height=320, use_container_width=True)

        # ------ ROSCA de Qualidade (Bom vs Refugo) ------
        with c2:
            st.markdown("**Qualidade: Bom vs Refugo (total)**")
            produzido_total = float(pd.to_numeric(oee_df["QUANTIDADE_PRODUTO"], errors="coerce").fillna(0).sum())
            refugo_total = float(pd.to_numeric(oee_df["QTD_REFUGADA"], errors="coerce").fillna(0).sum())
            bom_total = max(produzido_total - refugo_total, 0)
            pie_df = pd.DataFrame({"Categoria": ["Bom", "Refugo"], "Qtd": [bom_total, refugo_total]})
            if px:
                fig = px.pie(pie_df, values="Qtd", names="Categoria", hole=0.6)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(pie_df)

        # ------ Produção por Máquina (se houver alguma coluna de máquina) ------
        possiveis_maquina = [c for c in oee_df.columns if norm(c) in [norm(x) for x in ["mâq.", "maq", "maquina", "maquinas", "maquina_id"]]]
        if possiveis_maquina:
            colmaq = possiveis_maquina[0]
            st.markdown(f"**Produção por máquina** _(usando coluna: `{colmaq}`)_")
            prod_maquina = (
                oee_df.groupby(colmaq)["QUANTIDADE_PRODUTO"].sum().sort_values(ascending=False).reset_index()
            )
            if px:
                fig = px.bar(prod_maquina, x=colmaq, y="QUANTIDADE_PRODUTO")
                fig.update_layout(height=320, xaxis_title="Máquina", yaxis_title="Produzido")
                st.plotly_chart(fig, use_container_width=True)
            else:
                prod_maquina = prod_maquina.set_index(colmaq)
                st.bar_chart(prod_maquina, height=320, use_container_width=True)
        else:
            st.info("Não encontrei coluna de **máquina** no seu CSV. Se existir (ex.: 'MÁQ.'), ela aparece aqui.")

        # ------ Tendência mensal do OEE ------
        st.markdown("**Tendência mensal do OEE**")
        tmp = oee_df.copy()
        tmp["DATA_INICIO"] = pd.to_datetime(tmp["DATA_INICIO"], errors="coerce")
        tmp = tmp.dropna(subset=["DATA_INICIO"])
        if tmp.empty:
            st.info("Sem datas válidas suficientes para tendência.")
        else:
            tmp["YYYY-MM"] = tmp["DATA_INICIO"].dt.to_period("M").astype(str)
            oee_mensal = tmp.groupby("YYYY-MM")["OEE"].mean().reset_index()
            if px:
                fig = px.line(oee_mensal, x="YYYY-MM", y="OEE", markers=True)
                fig.update_layout(height=260, xaxis_title="Mês", yaxis_title="OEE")
                st.plotly_chart(fig, use_container_width=True)
            else:
                oee_mensal = oee_mensal.set_index("YYYY-MM")
                st.line_chart(oee_mensal, height=260, use_container_width=True)

# ======================== LOTE (.zip) =======================
with tab_batch:
    if uploaded_zip is None:
        st.info("Envie um **.zip** com vários CSVs para processar em lote.")
    else:
        try:
            z = zipfile.ZipFile(uploaded_zip)
            dfs_clean = []
            dfs_oee = []
            errors = []
            for name in z.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                try:
                    df_i = try_read_csv_bytes(z.read(name))
                    auto_i = auto_map_columns(df_i)
                    # tenta cache global (não tem fingerprint aqui)
                    final_i = {k: auto_i.get(k) for k in (REQUIRED + OPTIONAL)}
                    clean_i = build_clean(df_i, final_i)
                    clean_i["__ARQUIVO__"] = os.path.basename(name)
                    dfs_clean.append(clean_i)

                    oee_i = compute_oee(clean_i)
                    oee_i["__ARQUIVO__"] = os.path.basename(name)
                    dfs_oee.append(oee_i)
                except Exception as e:
                    errors.append(f"{name}: {e}")

            if dfs_clean:
                big_clean = pd.concat(dfs_clean, ignore_index=True)
                big_oee   = pd.concat(dfs_oee, ignore_index=True)

                st.success(f"Arquivos processados: {len(dfs_clean)}")
                st.dataframe(big_clean.head(30), use_container_width=True)

                st.download_button(
                    "⬇️ Baixar CSV LIMPO (lote)",
                    data=big_clean.to_csv(index=False).encode("utf-8-sig"),
                    file_name="easy_textil_limpo_lote.csv",
                )
                st.download_button(
                    "⬇️ Baixar OEE (lote)",
                    data=big_oee.to_csv(index=False).encode("utf-8-sig"),
                    file_name="easy_textil_oee_lote.csv",
                )
            else:
                st.warning("Nenhum CSV válido encontrado no .zip.")

            if errors:
                with st.expander("⚠️ Erros em alguns arquivos"):
                    st.write("\n".join(errors))
        except Exception as e:
            st.error(f"Erro ao ler .zip: {e}")

# ==================== GOOGLE SHEETS (opcional) ==============
with tab_sheets:
    st.subheader("🟢 Google Sheets (opcional)")

    if gspread is None:
        st.info("Para ativar esta aba instale `gspread` e `gspread_dataframe` no **requirements.txt**.")
        st.code("gspread\ngspread_dataframe", language="bash")
    else:
        sa_file = st.file_uploader("🔐 Suba a credencial **Service Account JSON**", type=["json"], key="gs_sa")
        plan_url_or_name = st.text_input("📄 URL da Planilha (ou Nome exato)", value="")
        aba_clean = st.text_input("🧼 Nome da aba p/ Dados Limpos", value="EasyTextil_Limpo")
        aba_oee = st.text_input("📊 Nome da aba p/ OEE", value="EasyTextil_OEE")

        colA, colB, colC = st.columns(3)
        with colA:
            ler_clean = st.button("📥 Ler aba LIMPO")
        with colB:
            gravar_clean = st.button("📤 Gravar LIMPO")
        with colC:
            gravar_oee = st.button("📤 Gravar OEE")

        if sa_file and plan_url_or_name:
            try:
                creds = json.load(sa_file)
                gc = gspread.service_account_from_dict(creds)
                sh = gc.open_by_url(plan_url_or_name) if plan_url_or_name.startswith("http") else gc.open(plan_url_or_name)
                st.success("✅ Conectado ao Google Sheets.")

                if ler_clean:
                    ws = sh.worksheet(aba_clean) if aba_clean in [w.title for w in sh.worksheets()] else sh.add_worksheet(aba_clean, rows=1000, cols=26)
                    df_in = get_as_dataframe(ws).dropna(how="all").dropna(axis=1, how="all")
                    if df_in.empty:
                        st.info("A aba está vazia.")
                    else:
                        st.dataframe(df_in.head(100), use_container_width=True)

                if gravar_clean:
                    if clean_df is None:
                        st.warning("Gere **Dados limpos** antes.")
                    else:
                        ws = sh.worksheet(aba_clean) if aba_clean in [w.title for w in sh.worksheets()] else sh.add_worksheet(aba_clean, rows=1000, cols=26)
                        ws.clear()
                        set_with_dataframe(ws, clean_df.reset_index(drop=True))
                        st.success("✅ Dados limpos enviados para o Google Sheets.")

                if gravar_oee:
                    if oee_df is None:
                        st.warning("Calcule o **OEE** antes.")
                    else:
                        ws = sh.worksheet(aba_oee) if aba_oee in [w.title for w in sh.worksheets()] else sh.add_worksheet(aba_oee, rows=1000, cols=26)
                        ws.clear()
                        set_with_dataframe(ws, oee_df.reset_index(drop=True))
                        st.success("✅ OEE enviado para o Google Sheets.")

            except Exception as e:
                st.error(f"Erro ao conectar/operar no Google Sheets: {e}")
        else:
            st.info("Envie a **credencial JSON** e preencha a **Planilha** para habilitar os botões.")

# ======================= DOWNLOADS ==========================
with tab_dl:
    if clean_df is not None:
        st.download_button(
            "⬇️ Baixar CSV LIMPO",
            data=clean_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="easy_textil_limpo.csv",
            type="primary",
        )
    if oee_df is not None:
        st.download_button(
            "⬇️ Baixar OEE (completo)",
            data=oee_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="easy_textil_oee.csv",
        )

# =================== DEDICATÓRIA / RODAPÉ ===================
st.markdown(
    """
    <div style="text-align:center; margin-top:1.25rem; color:#64748B;">
      <b>Easy Textil</b> — desenvolvido com carinho <br/>
      <span style="font-size:12px;">by <b>Reistec Tecnologia</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)
