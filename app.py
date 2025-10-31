# app.py
# Easy Textil — CSV normalizador com OEE, mapeamento assistido, batch .zip e logo na barra lateral
# Logo: assets/logo.jpg | Hero opcional: assets/hero.jpg

import io
import re
import os
import json
import zipfile
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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
# CSS SUAVE
# ============================================================
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; max-width: 1400px;}
h1 { font-weight: 800; letter-spacing: -0.5px; }
.small-muted { color:#6b7280; font-size:0.92rem; }
.chip { display:inline-flex; align-items:center; gap:.4rem; padding:.20rem .55rem;
        border-radius:999px; font-size:.78rem; font-weight:600; margin-right:.35rem; }
.chip-ok { background:#DCFCE7; color:#065F46; border:1px solid #86EFAC; }
.chip-warn { background:#FEF9C3; color:#854D0E; border:1px solid #FDE68A; }
.chip-bad { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
.card { border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; background:white; }
.card h3 { margin:0; font-size:0.95rem; color:#6b7280; font-weight:600; }
.card .value { font-size:1.35rem; font-weight:800; margin-top:.2rem; }
.hero { border:1px solid #e5e7eb; border-radius:16px; padding:18px; background:#F8FAFC; }
.hero h1 { margin:0; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR (logo + passo a passo + uploads)
# ============================================================
st.sidebar.image(logo_image, use_container_width=True)
st.sidebar.caption("Seu medidor de eficiência")

with st.sidebar.expander("📋 Passo a passo (bem simples)", expanded=True):
    st.markdown(
        """
1) **Baixe o CSV Modelo**.  
2) **Carregue um CSV** (ou **um .zip** com vários CSVs).  
3) **Confira o mapeamento** (aba *Mapeamento*).  
4) Veja **Dados Limpos** e **Insights/OEE**.  
5) **Exporte** os resultados.
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
    # campos auxiliares opcionais (para OEE ainda melhor)
    "TURNOS_NECESSARIOS": [1],
    "QUANTIDADE_PLANEJADA": [1000],    # ou KG_PLANEJADO
})
st.sidebar.download_button(
    "⬇️ Baixar CSV Modelo",
    data=MODEL_DF.to_csv(index=False).encode("utf-8-sig"),
    file_name="easy_textil_modelo.csv",
)

uploaded_csv = st.sidebar.file_uploader("📥 CSV único", type=["csv"], key="file_csv")
uploaded_zip = st.sidebar.file_uploader("📦 Lote (.zip com vários CSVs)", type=["zip"], key="file_zip")

# ============================================================
# ALIASES / COLUNAS
# ============================================================
REQUIRED = [
    "CLIENTE",
    "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS",
    "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN",
    "QTD_REFUGADA",
]
# auxiliares para OEE
OPTIONAL = [
    "TURNOS_NECESSARIOS",        # usado para tempo planejado (8h por turno)
    "QUANTIDADE_PLANEJADA",      # performance (Produzido / Planejado)
    "KG_PLANEJADO",              # sinônimo do acima
    "KG_PRODUZIDO",              # se vier, pode ser usado como QUANTIDADE_PRODUTO
    "HORAS_MAQ",                 # horas máquina efetivas (se existir)
]

ALIASES: Dict[str, List[str]] = {
    "CLIENTE": ["cliente", "pedido", "cliente/pedido"],
    "QUANTIDADE_PRODUTO": ["metros", "quantidade", "kg prod.", "kg produzido", "kg", "kg produzidos", "kg_produzido"],
    "MAQUINAS_NECESSARIAS": ["maquinas necessarias", "maq.", "maquinas", "turnos", "turnos necessarios"],
    "DATA_INICIO": ["inicio prod.", "inicio producao", "data inicio", "inicio"],
    "TEMPO_PARADA_MAQUINA_MIN": ["parada (min)", "parada", "tempo parada", "minutos parada"],
    "QTD_REFUGADA": ["kg rest.", "refugo", "kg refugado", "qtd refugado"],
    # opcionais
    "TURNOS_NECESSARIOS": ["turnos necessarios", "turnos"],
    "QUANTIDADE_PLANEJADA": ["quantidade planejada", "planejado", "kg. plan.", "kg plan", "kg planejado"],
    "KG_PLANEJADO": ["kg. plan.", "kg planejado", "planejado kg"],
    "KG_PRODUZIDO": ["kg prod.", "kg produzido", "kg produzidos"],
    "HORAS_MAQ": ["horas maq.", "horas maquina", "h maquina"],
}

PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12
}

def norm(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def parse_number(x) -> Optional[float]:
    if pd.isna(x): return pd.NA
    s = str(x).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA

def parse_date_ptbr(x) -> Optional[pd.Timestamp]:
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if s in ("", "**"): return pd.NaT
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.notna(dt): return dt
    m = re.match(r"(\d{1,2})[-/ ]([A-Za-z]{3,})\.?[-/ ]?(\d{2,4})?$", s)
    if m:
        d = int(m.group(1))
        mon = PT_MONTHS.get(m.group(2)[:3].lower())
        ytxt = m.group(3)
        if not ytxt: y = datetime.now().year
        elif len(ytxt) == 2: y = 2000 + int(ytxt)
        else: y = int(ytxt)
        try:
            return pd.Timestamp(year=y, month=mon, day=d)
        except Exception:
            return pd.NaT
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def extract_cliente_from_pedido(series: pd.Series) -> pd.Series:
    def _extract(s):
        if pd.isna(s): return pd.NA
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

def build_clean(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    clean = pd.DataFrame(index=df.index)

    # CLIENTE
    if mapping.get("CLIENTE"):
        clean["CLIENTE"] = df[mapping["CLIENTE"]].astype(str)
    else:
        ped_col = next((c for c in df.columns if "pedido" in norm(c)), None)
        clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]) if ped_col else pd.NA

    # QUANTIDADE_PRODUTO
    if mapping.get("KG_PRODUZIDO"):
        clean["QUANTIDADE_PRODUTO"] = df[mapping["KG_PRODUZIDO"]].apply(parse_number)
    elif mapping.get("QUANTIDADE_PRODUTO"):
        clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
    else:
        qtd_col = next((c for c in df.columns if "metro" in norm(c)), None)
        clean["QUANTIDADE_PRODUTO"] = df[qtd_col].apply(parse_number) if qtd_col else pd.NA

    # MAQUINAS_NECESSARIAS
    if mapping.get("MAQUINAS_NECESSARIAS"):
        clean["MAQUINAS_NECESSARIAS"] = (
            df[mapping["MAQUINAS_NECESSARIAS"]].apply(parse_number).fillna(1).astype("Int64")
        )
    else:
        t_col = next((c for c in df.columns if "turno" in norm(c)), None)
        if t_col:
            clean["MAQUINAS_NECESSARIAS"] = df[t_col].apply(parse_number).fillna(1).astype("Int64")
        else:
            clean["MAQUINAS_NECESSARIAS"] = 1

    # DATA_INICIO
    if mapping.get("DATA_INICIO"):
        clean["DATA_INICIO"] = df[mapping["DATA_INICIO"]].apply(parse_date_ptbr)
    else:
        candidate = next((c for c in df.columns if "inicio" in norm(c)), None)
        clean["DATA_INICIO"] = df[candidate].apply(parse_date_ptbr) if candidate else pd.NaT

    # TEMPO_PARADA_MAQUINA_MIN
    if mapping.get("TEMPO_PARADA_MAQUINA_MIN"):
        minutes = df[mapping["TEMPO_PARADA_MAQUINA_MIN"]].apply(parse_number)
        clean["TEMPO_PARADA_MAQUINA_MIN"] = minutes.fillna(0)
    else:
        # fallback: se existir "horas maq.", converte pra min
        h_col = mapping.get("HORAS_MAQ") or next(
            (c for c in df.columns if "hora" in norm(c) and "maq" in norm(c)), None
        )
        if h_col:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = (
                df[h_col].apply(parse_number).apply(lambda h: pd.NA if pd.isna(h) else h * 60).fillna(0)
            )
        else:
            clean["TEMPO_PARADA_MAQUINA_MIN"] = 0

    # QTD_REFUGADA
    if mapping.get("QTD_REFUGADA"):
        clean["QTD_REFUGADA"] = df[mapping["QTD_REFUGADA"]].apply(parse_number).fillna(0)
    else:
        r_col = next((c for c in df.columns if "rest" in norm(c) or "refug" in norm(c)), None)
        clean["QTD_REFUGADA"] = df[r_col].apply(parse_number).fillna(0) if r_col else 0

    # Auxiliares úteis (se existirem)
    if mapping.get("TURNOS_NECESSARIOS"):
        clean["TURNOS_NECESSARIOS"] = df[mapping["TURNOS_NECESSARIOS"]].apply(parse_number).fillna(1)
    else:
        clean["TURNOS_NECESSARIOS"] = pd.NA

    if mapping.get("QUANTIDADE_PLANEJADA"):
        clean["QUANTIDADE_PLANEJADA"] = df[mapping["QUANTIDADE_PLANEJADA"]].apply(parse_number)
    elif mapping.get("KG_PLANEJADO"):
        clean["QUANTIDADE_PLANEJADA"] = df[mapping["KG_PLANEJADO"]].apply(parse_number)
    else:
        clean["QUANTIDADE_PLANEJADA"] = pd.NA

    if mapping.get("HORAS_MAQ"):
        clean["HORAS_MAQ"] = df[mapping["HORAS_MAQ"]].apply(parse_number)
    else:
        clean["HORAS_MAQ"] = pd.NA

    return clean

def compute_oee(clean: pd.DataFrame) -> pd.DataFrame:
    """
    OEE = Disponibilidade × Performance × Qualidade

    - Disponibilidade = (Tempo Planejado - Paradas) / Tempo Planejado
      Tempo Planejado (min) = TURNOS_NECESSARIOS * 8h * 60
      Se TURNOS_NECESSARIOS não existe, usa MAQUINAS_NECESSARIAS * 8h * 60.
    - Performance = Produzido / Planejado (se houver). Caso contrário, 1.
    - Qualidade = (Produzido - Refugo) / Produzido  (se Produzido>0)
    """
    df = clean.copy()

    # Tempo planejado
    planned_by_turnos = pd.to_numeric(df.get("TURNOS_NECESSARIOS"), errors="coerce") * (8 * 60)
    planned_by_maqs = pd.to_numeric(df.get("MAQUINAS_NECESSARIAS"), errors="coerce") * (8 * 60)
    tempo_planejado = planned_by_turnos.fillna(planned_by_maqs).fillna(8 * 60)  # fallback 1 turno

    paradas = pd.to_numeric(df["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").fillna(0)
    tempo_operacao = (tempo_planejado - paradas).clip(lower=0)

    disponibilidade = (tempo_operacao / tempo_planejado).replace([pd.NA, pd.NaT], 0).fillna(0)
    disponibilidade = disponibilidade.clip(lower=0, upper=1)

    produzido = pd.to_numeric(df["QUANTIDADE_PRODUTO"], errors="coerce").fillna(0)
    planejado = pd.to_numeric(df.get("QUANTIDADE_PLANEJADA"), errors="coerce").fillna(pd.NA)

    performance = pd.Series(1.0, index=df.index)
    perf_mask = planejado.notna() & (planejado > 0)
    performance.loc[perf_mask] = (produzido.loc[perf_mask] / planejado.loc[perf_mask]).clip(lower=0)
    performance = performance.clip(upper=2.0)  # evita explosões absurdas

    refugado = pd.to_numeric(df["QTD_REFUGADA"], errors="coerce").fillna(0)
    qualidade = pd.Series(1.0, index=df.index)
    q_mask = (produzido > 0)
    qualidade.loc[q_mask] = ((produzido.loc[q_mask] - refugado.loc[q_mask]).clip(lower=0) / produzido.loc[q_mask])
    qualidade = qualidade.clip(lower=0, upper=1)

    oee = (disponibilidade * performance * qualidade).clip(lower=0)

    out = df.copy()
    out["TEMPO_PLANEJADO_MIN"] = tempo_planejado
    out["TEMPO_OPERACAO_MIN"] = tempo_operacao
    out["DISPONIBILIDADE"] = disponibilidade
    out["PERFORMANCE"] = performance
    out["QUALIDADE"] = qualidade
    out["OEE"] = oee

    return out

def mapping_status(mapping: Dict[str, Optional[str]]) -> str:
    chips = []
    for k in REQUIRED:
        if mapping.get(k):
            chips.append(f'<span class="chip chip-ok">✓ {k}</span>')
        else:
            chips.append(f'<span class="chip chip-bad">• {k} faltando</span>')
    return " ".join(chips)

# Persistência da sessão (mapeamento)
if "mapping_manual" not in st.session_state:
    st.session_state.mapping_manual = {}

# ============================================================
# HERO NO TOPO (logo fica só na sidebar)
# ============================================================
if hero_image:
    st.image(hero_image, use_container_width=True)
else:
    st.markdown(
        """
<div class="hero">
  <h1>Easy Textil — Seu medidor de eficiência</h1>
  <div class="small-muted">OEE = <b>Disponibilidade × Performance × Qualidade</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# TABS
# ============================================================
tab_prev, tab_map, tab_clean, tab_oee, tab_batch, tab_dl = st.tabs(
    ["🔎 Pré-visualização", "🧭 Mapeamento", "🧼 Dados limpos", "📊 OEE", "📦 Lote (.zip)", "⬇️ Downloads"]
)

# ================== ENTRADA CSV ÚNICO ======================
if uploaded_csv is None and uploaded_zip is None:
    with tab_prev:
        st.info("👈 Carregue um **CSV** ou um **.zip** com muitos CSVs na barra lateral.")
    st.stop()

# Lê CSV único (se houver)
df_raw = None
if uploaded_csv is not None:
    content = uploaded_csv.read()
    df_raw = try_read_csv_bytes(content)

# Mostra preview se tiver CSV único
with tab_prev:
    if df_raw is not None:
        st.markdown("### 📄 Pré-visualização do CSV")
        st.dataframe(df_raw.head(30), use_container_width=True)
    else:
        st.info("Carregue um CSV para visualizar aqui.")

# ================== MAPEAMENTO ==============================
with tab_map:
    if df_raw is None:
        st.info("Carregue um CSV para configurar o mapeamento.")
    else:
        auto_map = auto_map_columns(df_raw)

        st.subheader("🧭 Confirme/ajuste o mapeamento")
        st.caption("O sistema sugere automaticamente; ajuste se necessário e clique em **Aplicar**.")

        cols = list(df_raw.columns)
        pickers = {}
        grid1 = st.columns(3)
        grid2 = st.columns(3)
        keys = REQUIRED + [k for k in OPTIONAL if k not in REQUIRED]

        for i, key in enumerate(keys):
            target = ["— (não usar) —"] + cols
            default = st.session_state.mapping_manual.get(key) or auto_map.get(key)
            default = default if default in target else "— (não usar) —"
            container = grid1[i] if i < 3 else grid2[i-3 if i < 6 else 0] if i < 6 else st.container()
            pickers[key] = (container.selectbox if hasattr(container, "selectbox") else st.selectbox)(
                key,
                options=target,
                index=target.index(default) if default in target else 0,
                key=f"map_{key}",
                label=key
            )

        if st.button("✅ Aplicar mapeamento", type="primary"):
            st.session_state.mapping_manual = {
                k: (None if v == "— (não usar) —" else v) for k, v in pickers.items()
            }
            st.success("Mapeamento salvo nesta sessão.")

        # Salvar/abrir JSON do mapeamento
        c1, c2 = st.columns(2)
        with c1:
            mapping_json = json.dumps(st.session_state.mapping_manual or auto_map, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 Baixar mapeamento (JSON)",
                data=mapping_json.encode("utf-8"),
                file_name="mapeamento_easy_textil.json",
            )
        with c2:
            up_map = st.file_uploader("📂 Importar mapeamento (JSON)", type=["json"])
            if up_map is not None:
                try:
                    loaded = json.load(up_map)
                    if isinstance(loaded, dict):
                        st.session_state.mapping_manual = loaded
                        st.success("Mapeamento importado.")
                    else:
                        st.warning("JSON inválido.")
                except Exception as e:
                    st.error(f"Erro ao ler JSON: {e}")

        final_map = {k: st.session_state.mapping_manual.get(k) or auto_map.get(k) for k in (REQUIRED + OPTIONAL)}
        st.markdown("#### Status")
        st.markdown(mapping_status(final_map), unsafe_allow_html=True)

# ================== DADOS LIMPOS ============================
clean_df = None
if df_raw is not None:
    final_map = {k: st.session_state.mapping_manual.get(k) or auto_map_columns(df_raw).get(k)
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
            clientes_ok = clean_df["CLIENTE"].notna().sum()
            st.markdown(f'<div class="card"><h3>Clientes (preenchidos)</h3><div class="value">{clientes_ok:,}</div></div>', unsafe_allow_html=True)
        with c3:
            media_parada = float(pd.to_numeric(clean_df["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").fillna(0).mean())
            st.markdown(f'<div class="card"><h3>Média parada (min)</h3><div class="value">{media_parada:.1f}</div></div>', unsafe_allow_html=True)
        with c4:
            ref_total = float(pd.to_numeric(clean_df["QTD_REFUGADA"], errors="coerce").fillna(0).sum())
            st.markdown(f'<div class="card"><h3>Refugo total</h3><div class="value">{ref_total:.1f}</div></div>', unsafe_allow_html=True)

        # ✅ aqui usamos dataframe direto (sem HTML) — corrige o erro mostrado no print
        st.dataframe(clean_df.head(50), use_container_width=True)

# ================== OEE ============================
oee_df = None
with tab_oee:
    if clean_df is None:
        st.info("Gere os **Dados limpos** primeiro para calcular o OEE.")
    else:
        oee_df = compute_oee(clean_df)

        # Cards
        d_mean = float(oee_df["DISPONIBILIDADE"].mean(skipna=True))
        p_mean = float(oee_df["PERFORMANCE"].mean(skipna=True))
        q_mean = float(oee_df["QUALIDADE"].mean(skipna=True))
        o_mean = float(oee_df["OEE"].mean(skipna=True))

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="card"><h3>Disponibilidade</h3><div class="value">{d_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="card"><h3>Performance</h3><div class="value">{p_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="card"><h3>Qualidade</h3><div class="value">{q_mean:0.2%}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="card"><h3>OEE Médio</h3><div class="value">{o_mean:0.2%}</div></div>', unsafe_allow_html=True)

        # Série temporal (se houver datas)
        plot_df = oee_df.copy()
        plot_df["DATA_INICIO"] = pd.to_datetime(plot_df["DATA_INICIO"], errors="coerce")
        plot = plot_df.dropna(subset=["DATA_INICIO"]).groupby(pd.Grouper(key="DATA_INICIO", freq="D"))["OEE"].mean()
        if plot.dropna().empty:
            st.info("Sem datas válidas suficientes para gráfico.")
        else:
            st.line_chart(plot)

        st.markdown("#### Tabela OEE")
        st.dataframe(oee_df, use_container_width=True)

# ================== LOTE (.zip) ============================
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
                    final_i = {k: st.session_state.mapping_manual.get(k) or auto_i.get(k) for k in (REQUIRED + OPTIONAL)}
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
                big_oee = pd.concat(dfs_oee, ignore_index=True)

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

# ================== DOWNLOADS GERAIS =======================
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
