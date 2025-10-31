# app.py
# Easy Textil — Seu medidor de eficiência (carregador/normalizador de CSV)
# Compatível com CSVs no formato do exemplo enviado (colunas pt-BR, datas e números mistos)

import io
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Easy Textil — Seu medidor de eficiência", layout="wide")

# ============================================================
# Configurações & dicionários
# ============================================================

REQUIRED = [
    "CLIENTE",
    "QUANTIDADE_PRODUTO",
    "MAQUINAS_NECESSARIAS",
    "DATA_INICIO",
    "TEMPO_PARADA_MAQUINA_MIN",
    "QTD_REFUGADA",
]

# Sinônimos / aliases comuns encontrados em planilhas têxteis
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
# Utilidades
# ============================================================

def norm(s: str) -> str:
    """Normaliza cabeçalho: sem acentos, minúsculo, espaços/pts/underscores uniformes."""
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
    """Converte strings como '11.500' (milhar) e '23,8' (decimal) para float.
       '**', '', None → NaN"""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s == "**":
        return pd.NA
    # Remove espaços
    s = s.replace(" ", "")
    # Se tem vírgula como decimal, trocamos por ponto; pontos viram separador de milhar → removidos
    # Ex.: 11.500 -> 11500 ; 23,8 -> 23.8 ; 1.234,56 -> 1234.56
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
    """Aceita '12-dez.', '09/jan.', '7-jan.', '23/jan', '24/jan.', '23-jan-24', '24/01/2025', etc."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s == "" or s == "**":
        return pd.NaT

    # Tenta direto com pandas (muitas vezes basta)
    try:
        # dayfirst=True já ajuda bastante com formatos brasileiros
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt
    except Exception:
        pass

    # Normaliza: remove pontinhos depois do mês (jan., fev., etc.)
    s2 = re.sub(r"([A-Za-z]{3})\.", r"\1", s, flags=re.IGNORECASE)

    # Formatos tipo "7-jan." ou "7-jan" ou "07-jan-24"
    m = re.match(r"^\s*(\d{1,2})\s*[-/ ]\s*([A-Za-z]{3,})\s*[-/ ]?\s*(\d{2,4})?\s*$", s2, flags=re.IGNORECASE)
    if m:
        d = int(m.group(1))
        mon = _pt_month_to_num(m.group(2))
        y = m.group(3)
        if mon is not None:
            if y is None:
                # Sem ano → heurística: usa ano corrente
                y = datetime.now().year
            else:
                y = int(y)
                if y < 100:
                    y += 2000  # 24 -> 2024
            try:
                return pd.Timestamp(year=y, month=mon, day=d)
            except Exception:
                return pd.NaT

    # Última tentativa
    try:
        return pd.to_datetime(s2, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def try_read_csv(file) -> pd.DataFrame:
    """Le o CSV tentando variações de encoding e separadores (, ; \t)."""
    content = file.read()
    for sep in [",", ";", "\t", "|"]:
        for enc in ["utf-8-sig", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc, engine="python")
                if df.shape[1] > 1 or len(df) == 0:
                    return df
            except Exception:
                continue
    # fallback: deixa o pandas decidir
    return pd.read_csv(io.BytesIO(content), engine="python")

def suggest_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """Sugere um mapeamento automático com base em ALIASES."""
    normalized = {c: norm(c) for c in columns}
    inv = {v: k for k, v in normalized.items()}  # norm_col_name -> original col

    mapping = {k: None for k in REQUIRED}

    # 1) Casos óbvios: busca por sinônimos
    for target, names in ALIASES.items():
        found = None
        for cand in names:
            n = norm(cand)
            # tenta match exato pelo nome normalizado existente
            if n in inv:
                found = inv[n]
                break
            # tenta "startswith" (ex.: 'kg prod' casar com 'kg prod.')
            for nc, orig in inv.items():
                if nc.startswith(n) or n in nc:
                    found = orig
                    break
            if found:
                break
        mapping[target] = found

    # 2) Ajustes inteligentes específicos
    # CLIENTE: se mapeou para 'pedido', vamos extrair o nome depois do código
    # QUANTIDADE_PRODUTO: se houver 'kg prod.' e 'metros', preferimos 'kg prod.'; senão 'metros'
    if mapping["QUANTIDADE_PRODUTO"] is None:
        # tenta heurísticas comuns
        for pref in ["Kg PROD.", "Kg Prod.", "Kg PROD", "Kg", "METROS", "Quant.", "Quantidade", "QTD"]:
            for c in columns:
                if norm(c) == norm(pref):
                    mapping["QUANTIDADE_PRODUTO"] = c
                    break
            if mapping["QUANTIDADE_PRODUTO"]:
                break

    return mapping

def extract_cliente_from_pedido(series: pd.Series) -> pd.Series:
    """
    Ex.: '98406 JK INDUSTRIA' -> 'JK INDUSTRIA'
         '98388 AZZURRA'     -> 'AZZURRA'
    Se não encontrar padrão, retorna o texto original.
    """
    def _extract(s):
        if pd.isna(s):
            return pd.NA
        s = str(s).strip()
        m = re.match(r"^\s*\d+\s+(.*)$", s)
        return m.group(1).strip() if m else s
    return series.apply(_extract)

def to_minutes_from_hours(series: pd.Series) -> pd.Series:
    """Converte horas (float) em minutos."""
    return series.apply(lambda v: (float(v) * 60) if pd.notna(v) else pd.NA)

def coalesce(s: pd.Series, default):
    return s.fillna(default)

# ============================================================
# UI
# ============================================================

st.sidebar.image(
    "https://em-content.zobj.net/source/microsoft-teams/363/factory-worker_1f9d1-200d-1f3ed.png",
    width=72,
)
st.sidebar.title("Easy Textil")
st.sidebar.caption("Seu medidor de eficiência")

st.title("Easy Textil — Seu medidor de eficiência")
st.caption("OEE = **Disponibilidade × Performance × Qualidade**")

uploaded = st.sidebar.file_uploader("📥 Carregue um CSV", type=["csv"], help="Limite ~200MB • CSV")

# Botão para baixar um CSV modelo compatível
MODEL_DF = pd.DataFrame(
    {
        "CLIENTE": ["Exemplo Indústria"],
        "QUANTIDADE_PRODUTO": [1000],
        "MAQUINAS_NECESSARIAS": [1],
        "DATA_INICIO": ["2025-01-07"],
        "TEMPO_PARADA_MAQUINA_MIN": [0],
        "QTD_REFUGADA": [0],
        # Colunas opcionais comuns
        "PEDIDO": ["98406 Exemplo Indústria"],
        "METROS": [11500],
        "MÁQ.": ["Tear 01"],
        "HORAS MAQ.": [0.0],
        "Kg PROD.": [980],
        "Kg REST.": [20],
    }
)
model_csv = MODEL_DF.to_csv(index=False).encode("utf-8-sig")
st.sidebar.download_button("⬇️ Baixar CSV modelo", data=model_csv, file_name="easy_textil_modelo.csv", mime="text/csv")

if uploaded is None:
    st.info("Carregue um CSV na barra lateral à esquerda. Dica: você pode **arrastar e soltar** o arquivo.")
    st.stop()

# ============================================================
# Leitura e normalização básica
# ============================================================

raw_df = try_read_csv(uploaded)
orig_cols = list(raw_df.columns)

if raw_df.empty:
    st.error("Não consegui ler dados nesse arquivo. Verifique o separador (`,` ou `;`) e o encoding.")
    st.stop()

st.subheader("Pré-visualização do arquivo bruto")
st.dataframe(raw_df.head(20), use_container_width=True)

# Sugestão de mapeamento
suggested = suggest_mapping(orig_cols)

st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Mapeamento de colunas")

mapping: Dict[str, Optional[str]] = {}
for target in REQUIRED:
    mapping[target] = st.sidebar.selectbox(
        f"{target}",
        options=[None] + orig_cols,
        index=( [None] + orig_cols ).index(suggested.get(target)) if suggested.get(target) in orig_cols else 0,
        help=f"Selecione a coluna do seu CSV que representa **{target}**.",
        format_func=lambda x: "— selecione —" if x is None else x,
        key=f"map_{target}"
    )

st.sidebar.markdown(
    """
    *Dicas de mapeamento*:
    - **CLIENTE**: se vier dentro de **PEDIDO**, eu extraio o nome (ex.: `98406 JK INDUSTRIA` → `JK INDUSTRIA`).
    - **QUANTIDADE_PRODUTO**: pode ser `METROS` ou `Kg PROD.` — escolha o que você usa para medir produção.
    - **TEMPO_PARADA_MAQUINA_MIN**: se você só tiver `HORAS MAQ.`, mapeie e eu converto para **minutos**.
    """
)

# ============================================================
# Construção do DataFrame "clean"
# ============================================================

df = raw_df.copy()

# Normaliza números em todas as colunas candidatas
for c in df.columns:
    # tenta converter números sem quebrar colunas textuais
    if df[c].dtype == object:
        # detecta rapidamente se coluna tem traços de número BR
        sample = str(df[c].dropna().astype(str).head(10).tolist())
        if re.search(r"(\d+[\.,]\d+)|(\d{1,3}\.\d{3})", sample) or re.search(r"^\d+$", sample):
            df[c] = df[c].apply(parse_number)

# Normaliza datas em possíveis colunas de data
for c in df.columns:
    if any(tok in norm(c) for tok in ["data", "inicio", "início", "fim", "entrega"]):
        # tenta parsear
        df[c] = df[c].apply(parse_date_ptbr)

clean = pd.DataFrame()

# CLIENTE
if mapping["CLIENTE"]:
    if norm(mapping["CLIENTE"]) == norm("pedido"):
        clean["CLIENTE"] = extract_cliente_from_pedido(df[mapping["CLIENTE"]])
    else:
        clean["CLIENTE"] = df[mapping["CLIENTE"]].astype("string")
else:
    # fallback: tenta PEDIDO
    ped_col = next((c for c in orig_cols if norm(c) == "pedido"), None)
    if ped_col:
        clean["CLIENTE"] = extract_cliente_from_pedido(df[ped_col]).astype("string")
    else:
        clean["CLIENTE"] = pd.Series(pd.NA, index=df.index, dtype="string")

# QUANTIDADE_PRODUTO
if mapping["QUANTIDADE_PRODUTO"]:
    clean["QUANTIDADE_PRODUTO"] = df[mapping["QUANTIDADE_PRODUTO"]].apply(parse_number)
else:
    # tenta METROS ou Kg PROD.
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
    # tenta TURNOS NECESSÁRIOS como proxy
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
    # Se usuário mapeou uma coluna de "horas", converte para minutos:
    if "hora" in norm(col):
        clean["TEMPO_PARADA_MAQUINA_MIN"] = to_minutes_from_hours(df[col])
    else:
        # já está em minutos ou é numérico genérico
        vals = df[col].apply(parse_number)
        clean["TEMPO_PARADA_MAQUINA_MIN"] = vals
else:
    # fallback: se houver HORAS MAQ., converte
    horas_col = next((c for c in orig_cols if "hora" in norm(c) and "maq" in norm(c)), None)
    if horas_col:
        clean["TEMPO_PARADA_MAQUINA_MIN"] = to_minutes_from_hours(df[horas_col].apply(parse_number))
    else:
        clean["TEMPO_PARADA_MAQUINA_MIN"] = pd.Series(0, index=df.index, dtype="Float64")

# QTD_REFUGADA
if mapping["QTD_REFUGADA"]:
    clean["QTD_REFUGADA"] = coalesce(df[mapping["QTD_REFUGADA"]].apply(parse_number), 0).astype("Float64")
else:
    # heurística: usar 'Kg REST.' se existir (valores negativos tratados como 0)
    rest_col = next((c for c in orig_cols if "kg" in norm(c) and "rest" in norm(c)), None)
    if rest_col:
        tmp = df[rest_col].apply(parse_number)
        # se número negativo, assume que é ajuste e não refugo
        tmp = tmp.apply(lambda v: max(v, 0) if pd.notna(v) else v)
        clean["QTD_REFUGADA"] = coalesce(tmp, 0).astype("Float64")
    else:
        clean["QTD_REFUGADA"] = pd.Series(0, index=df.index, dtype="Float64")

# ============================================================
# Validação e feedback
# ============================================================

missing = [c for c in REQUIRED if c not in clean.columns or clean[c].isna().all()]
if missing:
    st.error(
        "Colunas obrigatórias ausentes ou vazias: **{}**. "
        "Ajuste o mapeamento na barra lateral. Dica: você também pode baixar o CSV modelo."
        .format(", ".join(missing))
    )
else:
    st.success("✅ Colunas obrigatórias mapeadas com sucesso!")

# Preview do dataset limpo
st.subheader("Dados limpos (prontos para o cálculo de OEE)")
st.dataframe(clean.head(50), use_container_width=True)

# Exportar CSV limpo
out_csv = clean.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV LIMPO (compatível)", data=out_csv, file_name="easy_textil_limpo.csv", mime="text/csv")

# ============================================================
# (Opcional) Cálculos básicos para OEE - placeholders
# Aqui você pode integrar com seu cálculo existente, usando as colunas já padronizadas.
# ============================================================

with st.expander("🧮 (Opcional) Exemplo de métricas rápidas"):
    # Exemplos simples (ajuste conforme seu modelo real)
    total_prod = pd.to_numeric(clean["QUANTIDADE_PRODUTO"], errors="coerce").sum(min_count=1)
    total_ref = pd.to_numeric(clean["QTD_REFUGADA"], errors="coerce").sum(min_count=1)
    tempo_parada_min = pd.to_numeric(clean["TEMPO_PARADA_MAQUINA_MIN"], errors="coerce").sum(min_count=1)
    maquinas_media = pd.to_numeric(clean["MAQUINAS_NECESSARIAS"], errors="coerce").mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quant. Produzida (soma)", f"{total_prod:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if pd.notna(total_prod) else "—")
    col2.metric("Refugo (soma)", f"{total_ref:,.2f}".replace(",", "X").replace(".", ",").replace("X", ",") if pd.notna(total_ref) else "—")
    col3.metric("Paradas (min, soma)", f"{tempo_parada_min:,.0f}".replace(",", ".") if pd.notna(tempo_parada_min) else "—")
    col4.metric("Máquinas necessárias (média)", f"{maquinas_media:.2f}".replace(".", ",") if pd.notna(maquinas_media) else "—")

st.caption("💡 Se quiser, posso integrar estes dados diretamente ao seu cálculo de OEE atual — é só me dizer onde ele está no seu projeto.")
