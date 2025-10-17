# Projeto — Contabilidade e Finanças
# Giovanna Mathias

import math
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import combinations

# ============ CONFIG ============
TICKERS = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']  # 3 empresas
MARKET_TICKER = '^BVSP'  # Ibovespa (Yahoo Finance)
HOJE = datetime.today().date()
START = (HOJE - timedelta(days=365*4 + 20)).isoformat()
END = HOJE.isoformat()
ORCAMENTO_TOTAL = 100000.00
MESES_ANO = 12

RISK_FREE_ANNUAL = 0.10

def rf_mensal(rf_a):
    # converte anualmente para mensal por capitalização composta
    return (1.0 + rf_a) ** (1.0 / MESES_ANO) - 1.0

# ============ DOWNLOAD (Adj Close) ============
print(f"Baixando dados de {TICKERS} e do mercado ({MARKET_TICKER}) entre {START} e {END}...")
precos_d = yf.download(TICKERS + [MARKET_TICKER], start=START, end=END,
                       auto_adjust=True, progress=False)['Close']

# Garantia de DataFrame
if isinstance(precos_d, pd.Series):
    precos_d = precos_d.to_frame(name=TICKERS[0])

precos_d = precos_d.dropna(how='all').sort_index()
# exige cotação de TODOS no dia (ativos + mercado)
precos_d = precos_d.dropna(how='any')

# AGREGA MENSAL: último preço ajustado de cada mês
precos_m = precos_d.resample('M').last().dropna(how='any')

# Separa colunas de ativos e mercado
ativos_cols = [c for c in precos_m.columns if c in TICKERS]
market_col = MARKET_TICKER

# ============ FUNÇÕES MANUAIS ============
def media(lst):
    return sum(lst) / len(lst) if lst else float('nan')

def desvio_padrao(lst):
    n = len(lst)
    if n < 2:
        return float('nan')
    m = media(lst)
    var = sum((x - m) ** 2 for x in lst) / (n - 1)   # amostral
    return math.sqrt(var)

def covariancia(x, y):
    n = len(x)
    if n != len(y) or n < 2:
        return float('nan')
    mx, my = media(x), media(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)

def correlacao(x, y):
    sx, sy = desvio_padrao(x), desvio_padrao(y)
    if sx == 0 or sy == 0 or math.isnan(sx) or math.isnan(sy):
        return float('nan')
    return covariancia(x, y) / (sx * sy)

def retornos_mensais(lista_precos):
    # R_m = P_m/P_{m-1} - 1  (Preço Ajustado já incorpora eventos)
    return [(lista_precos[i] / lista_precos[i-1]) - 1.0 for i in range(1, len(lista_precos))]

def retorno_acumulado(retornos):
    # ∏(1+R_m) - 1
    prod = 1.0
    for r in retornos:
        prod *= (1.0 + r)
    return prod - 1.0

def fmt_pct(x, casas=2):  # para imprimir em %
    return f"{x*100:.{casas}f}%"

def variancia(lst):
    n = len(lst)
    if n < 2:
        return float('nan')
    m = media(lst)
    return sum((x - m) ** 2 for x in lst) / (n - 1)

# ============ RETORNOS MENSAIS ============
ret_mensal = {}      # ticker -> lista de retornos mensais
estat = {}           # métricas por ativo

# Ativos
for t in ativos_cols:
    serie = precos_m[t].tolist()
    r = retornos_mensais(serie)
    ret_mensal[t] = r

# Mercado (Ibovespa)
serie_m = precos_m[market_col].tolist()
ret_mkt_m = retornos_mensais(serie_m)

# ============ ESTATÍSTICAS (ATIVOS) ============
for t in ativos_cols:
    r = ret_mensal[t]
    mu_m = media(r)
    sd_m = desvio_padrao(r)

    # anualização a partir de mensais
    mu_a = mu_m * MESES_ANO
    sd_a = sd_m * math.sqrt(MESES_ANO)
    cv_a = (sd_a / mu_a) if mu_a != 0 else float('nan')
    ret_acum_4a = retorno_acumulado(r)

    estat[t] = {
        'PrecoAtual': precos_m[t].iloc[-1],
        'RetornoMedio_m': mu_m,
        'DesvioPadrao_m': sd_m,
        'RetornoMedio_a': mu_a,
        'DesvioPadrao_a': sd_a,
        'CV_anual': cv_a,
        'RetAcum_4a': ret_acum_4a
    }

resumo_ativos = pd.DataFrame(estat).T

# ============ MERCADO: MÉDIAS ============
mu_mkt_m = media(ret_mkt_m)
mu_mkt_a = mu_mkt_m * MESES_ANO
var_mkt_m = variancia(ret_mkt_m)

# ============ BETAS (com Ibovespa) ============
betas = {}
for t in ativos_cols:
    cov_im = covariancia(ret_mensal[t], ret_mkt_m)
    beta = cov_im / var_mkt_m if not math.isnan(cov_im) and var_mkt_m not in (0.0, float('nan')) else float('nan')
    betas[t] = beta

betas_s = pd.Series(betas, name='Beta')

# ============ CAPM ============
rf_a = RISK_FREE_ANNUAL
rf_m = rf_mensal(rf_a)

# Prêmio de mercado (anual)
market_premium_a = mu_mkt_a - rf_a

capm_a = {}
capm_m = {}
for t in ativos_cols:
    b = betas[t]
    # CAPM anual e mensal
    capm_a[t] = rf_a + b * market_premium_a if not math.isnan(b) else float('nan')
    capm_m[t] = rf_m + b * (mu_mkt_m - rf_m) if not math.isnan(b) else float('nan')

capm_df = pd.DataFrame({
    'Beta': betas_s,
    'CAPM_mensal': pd.Series(capm_m),
    'CAPM_anual': pd.Series(capm_a)
})

# ============ MATRIZ DE CORRELAÇÃO (mensal) ============
N = len(ativos_cols)
corr_matrix = pd.DataFrame(index=ativos_cols, columns=ativos_cols, dtype=float)
for i in range(N):
    for j in range(N):
        x = ret_mensal[ativos_cols[i]]
        y = ret_mensal[ativos_cols[j]]
        corr_matrix.iloc[i, j] = 1.0 if i == j else correlacao(x, y)

print("\n=== MATRIZ DE CORRELAÇÃO (mensal) ===")
print(corr_matrix.round(4))

# ---- Correlação 2 a 2 (lista de pares)
from itertools import combinations
print("\n=== CORRELAÇÃO 2 a 2 (mensal) ===")
for a, b in combinations(ativos_cols, 2):
    rho = correlacao(ret_mensal[a], ret_mensal[b])
    print(f"{a}  x  {b}:  {rho:.3f}")

# ============ DETALHES POR AÇÃO ============
print("\n=== DETALHES POR AÇÃO (base MENSAL → anualizada) ===")
for t in ativos_cols:
    r = ret_mensal[t]
    n = len(r)
    mu_m = media(r)
    sd_m = desvio_padrao(r)
    mu_a = mu_m * MESES_ANO
    sd_a = sd_m * math.sqrt(MESES_ANO)
    cv_a = (sd_a / mu_a) if mu_a != 0 else float('nan')
    ret_acum = retorno_acumulado(r)
    preco_atual = float(precos_m[t].iloc[-1])
    r_min = min(r) if r else float('nan')
    r_max = max(r) if r else float('nan')

    print(f"\n[{t}]")
    print(f"Preço atual: R$ {preco_atual:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
    print(f"Observações (meses): {n}")
    print(f"Retorno médio mensal: {fmt_pct(mu_m)}")
    print(f"Volatilidade mensal: {fmt_pct(sd_m)}")
    print(f"Retorno médio anualizado: {fmt_pct(mu_a)}")
    print(f"Volatilidade anualizada: {fmt_pct(sd_a)}")
    print(f"CV (σ/μ, anual): {cv_a:.3f}x")
    print(f"Retorno acumulado no período: {fmt_pct(ret_acum)}")
    print(f"Melhor mês: {fmt_pct(r_max)} | Pior mês: {fmt_pct(r_min)}")

# ============ RANKINGS ============
print("\n=== RANKING — Retorno médio anual (maior → menor) ===")
print(resumo_ativos['RetornoMedio_a'].sort_values(ascending=False).apply(lambda v: f"{v*100:.2f}%"))

print("\n=== RANKING — CV anual (menor → melhor) ===")
print(resumo_ativos['CV_anual'].sort_values().apply(lambda v: f"{v:.3f}x"))

# ============ CARTEIRA (pesos iguais) — métricas ANUAIS ============
w = [1 / N] * N
mu_port_a = sum(w[i] * estat[ativos_cols[i]]['RetornoMedio_a'] for i in range(N))

# variância anual: Σ_i Σ_j w_i w_j * (cov_mensal * 12)
sig2_port_a = 0.0
for i in range(N):
    for j in range(N):
        cov_ij_m = covariancia(ret_mensal[ativos_cols[i]], ret_mkt_m if i == j and ativos_cols[i] == market_col else ret_mensal[ativos_cols[j]])
        # OBS: para a carteira, precisamos cov entre ativos (não com mercado).
        # Ajuste correto:
        cov_ij_m = covariancia(ret_mensal[ativos_cols[i]], ret_mensal[ativos_cols[j]])
        cov_ij_a = cov_ij_m * MESES_ANO
        sig2_port_a += w[i] * w[j] * cov_ij_a
sigma_port_a = math.sqrt(sig2_port_a)

# ============ ALOCAÇÃO REAL ============
ultimos_precos = precos_m[ativos_cols].iloc[-1]
qtd = ((ORCAMENTO_TOTAL / N) // ultimos_precos).astype(int)  # inteiras
investido = qtd * ultimos_precos
total_invest = float(investido.sum())
pesos_reais = (investido / total_invest)

# retorno esperado anual com pesos reais
mu_port_real_a = sum(pesos_reais[i] * estat[ativos_cols[i]]['RetornoMedio_a'] for i in range(N))

# ============ IMPRESSÕES (em %) ============
def linha_aloc():
    df = pd.DataFrame({'Qtd': qtd, 'Investido(R$)': investido.round(2), 'PesoReal': pesos_reais.round(4)})
    return df

print("\n=== RESUMO DA CARTEIRA (base mensal anualizada, em %) ===")
print(f"Retorno médio esperado (anual, pesos iguais): {fmt_pct(mu_port_a)}")
print(f"Volatilidade anual (pesos iguais): {fmt_pct(sigma_port_a)}")
print("\nAlocação com R$ 100.000 (quantidades inteiras):")
print(linha_aloc())
print(f"\nTotal investido: R$ {total_invest:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
print(f"Retorno médio esperado (anual, pesos reais): {fmt_pct(mu_port_real_a)}")

# ============ SAÍDAS CAPM ============
print("\n=== MERCADO (Ibovespa) ===")
print(f"Retorno médio do mercado (mensal): {fmt_pct(mu_mkt_m)}")
print(f"Retorno médio do mercado (anual):  {fmt_pct(mu_mkt_a)}")
print(f"RF mensal: {fmt_pct(rf_m)} | RF anual: {fmt_pct(rf_a)}")
print(f"Prêmio de mercado (anual): {fmt_pct(market_premium_a)}")

print("\n=== BETAS (em relação ao Ibovespa) ===")
for t in ativos_cols:
    print(f"{t}: beta = {betas[t]:.3f}")

print("\n=== CAPM (esperado) ===")
for t in ativos_cols:
    print(f"{t}: CAPM mensal = {fmt_pct(capm_m[t])} | CAPM anual = {fmt_pct(capm_a[t])}")

# ====================== INDICADORES FINANCEIROS (2021–2024) ======================
# Requer: yfinance >= 0.2.x. Funciona melhor quando o Yahoo expõe os campos para o ticker.
# Fallback: CSV manual caso algum dado não esteja disponível.

import os

ANOS_INDICADORES = [2021, 2022, 2023, 2024]

# ---- Mapeamentos de chaves possíveis no Yahoo (variam por empresa/idioma) ----
KEYS = {
    "ativo_circulante": [
        "Total Current Assets", "Current Assets", "Ativo Circulante"
    ],
    "passivo_circulante": [
        "Total Current Liabilities", "Current Liabilities", "Passivo Circulante"
    ],
    "estoques": [
        "Inventory", "Inventories", "Estoques"
    ],
    "fornecedores": [
        "Accounts Payable", "Trade and Other Payables", "Fornecedores"
    ],
    "ativo_total": [
        "Total Assets", "Ativo Total"
    ],
    "receita_liquida": [
        "Total Revenue", "Revenue", "Receita Líquida"
    ],
    "cmv": [
        "Cost Of Revenue", "Cost of Goods Sold", "COGS", "Custo das Mercadorias Vendidas", "Custo dos Produtos Vendidos"
    ]
}

def _first_key_available(df, candidates):
    for k in candidates:
        if k in df.index:
            return k
    return None

def _to_year(dt_like):
    # yfinance usa colunas datadas (Timestamp). Pega o ano.
    try:
        return int(str(dt_like)[:4])
    except:
        return None

def _extract_by_year(table, key_candidates, year):
    """
    table: DataFrame no padrão yfinance (linhas = contas, colunas = datas).
    Retorna valor da conta no ANO informado (preferindo colunas do próprio ano; senão, a mais próxima anterior).
    """
    if table is None or table.empty:
        return None

    key = _first_key_available(table, key_candidates)
    if not key:
        return None

    # Mapear colunas por ano
    cols_by_year = {}
    for c in table.columns:
        y = _to_year(c)
        if y:
            cols_by_year[y] = c

    if year in cols_by_year:
        return float(table.loc[key, cols_by_year[year]])
    else:
        # pegar a coluna mais próxima anterior ao ano (ex.: usar 2023 se não houver 2024 ainda)
        anos_disponiveis = sorted([y for y in cols_by_year.keys() if y <= year])
        if anos_disponiveis:
            return float(table.loc[key, cols_by_year[anos_disponiveis[-1]]])
    return None

def _try_fetch_from_yf(ticker, anos):
    """
    Tenta extrair dados do Yahoo para os anos pedidos.
    Retorna um dict {ano: {campos...}}, podendo conter None.
    """
    data = {ano: {} for ano in anos}
    try:
        tk = yf.Ticker(ticker)
        # balance_sheet e financials anuais
        bs = tk.balance_sheet    # Balanço Patrimonial (Annual)
        fin = tk.financials      # DRE (Annual)
        # Alguns tickers disponibilizam 'yearly_...' nas versões novas:
        # bs = tk.get_balance_sheet(freq="yearly")
        # fin = tk.get_financials(freq="yearly")

        for ano in anos:
            ac = _extract_by_year(bs, KEYS["ativo_circulante"], ano)
            pc = _extract_by_year(bs, KEYS["passivo_circulante"], ano)
            est = _extract_by_year(bs, KEYS["estoques"], ano)
            forn = _extract_by_year(bs, KEYS["fornecedores"], ano)
            at = _extract_by_year(bs, KEYS["ativo_total"], ano)
            rec = _extract_by_year(fin, KEYS["receita_liquida"], ano)
            cmv = _extract_by_year(fin, KEYS["cmv"], ano)

            data[ano] = {
                "AtivoCirculante": ac,
                "PassivoCirculante": pc,
                "Estoques": est,
                "Fornecedores": forn,
                "AtivoTotal": at,
                "ReceitaLiquida": rec,
                "CMV": cmv
            }
    except Exception as e:
        print(f"[{ticker}] Aviso: falha ao consultar yfinance ({e}).")
    return data

def _calc_indicadores(reg):
    """
    reg: dict com os campos numéricos.
    Retorna dict com LC, LS, GA e PMP (dias).
    """
    ac = reg.get("AtivoCirculante")
    pc = reg.get("PassivoCirculante")
    est = reg.get("Estoques")
    at = reg.get("AtivoTotal")
    rec = reg.get("ReceitaLiquida")
    cmv = reg.get("CMV")
    forn = reg.get("Fornecedores")

    def safe_div(num, den):
        if num is None or den in (None, 0):
            return None
        return num / den

    lc = safe_div(ac, pc)
    ls = safe_div((ac - est) if (ac is not None and est is not None) else None, pc)
    ga = safe_div(rec, at)
    pmp = safe_div(forn * 360 if forn is not None else None, cmv)

    return {
        "LiquidezCorrente": lc,
        "LiquidezSeca": ls,
        "GiroAtivo": ga,
        "PMP_dias": pmp
    }

# ---------- CSV de fallback (caso algum campo não venha do Yahoo) ----------
# Template (um arquivo por projeto; você pode preencher ou complementar valores faltantes):
CSV_TEMPLATE = "indicadores_input.csv"
if not os.path.exists(CSV_TEMPLATE):
    import pandas as pd
    cols = ["Ticker","Ano","AtivoCirculante","PassivoCirculante","Estoques","Fornecedores","AtivoTotal","ReceitaLiquida","CMV"]
    df_tmp = pd.DataFrame(columns=cols)
    df_tmp.to_csv(CSV_TEMPLATE, index=False, encoding="utf-8")
    print(f"\n[Template criado] Preencha valores faltantes em: {CSV_TEMPLATE}")
    print("Colunas: Ticker,Ano,AtivoCirculante,PassivoCirculante,Estoques,Fornecedores,AtivoTotal,ReceitaLiquida,CMV\n")

def _merge_with_csv(ticker, data_by_year, csv_path=CSV_TEMPLATE):
    """
    Se existir um CSV com valores preenchidos, ele sobrescreve/complete os dados do Yahoo.
    """
    if not os.path.exists(csv_path):
        return data_by_year
    try:
        df_csv = pd.read_csv(csv_path)
        df_csv = df_csv[df_csv["Ticker"].astype(str).str.upper() == str(ticker).upper()]
        for _, row in df_csv.iterrows():
            ano = int(row["Ano"])
            if ano in data_by_year:
                for col in ["AtivoCirculante","PassivoCirculante","Estoques","Fornecedores","AtivoTotal","ReceitaLiquida","CMV"]:
                    val = row.get(col, None)
                    if pd.notnull(val):
                        data_by_year[ano][col] = float(val)
    except Exception as e:
        print(f"[{ticker}] Aviso ao ler CSV fallback: {e}")
    return data_by_year

# ----------------- Execução por ticker: calcula e agrega numa tabela -----------------
linhas = []
for t in ativos_cols:
    print(f"\n[INDICADORES] Coletando dados contábeis para {t} (anos: {ANOS_INDICADORES})...")
    dados = _try_fetch_from_yf(t, ANOS_INDICADORES)
    dados = _merge_with_csv(t, dados, CSV_TEMPLATE)

    for ano in ANOS_INDICADORES:
        reg = dados.get(ano, {})
        inds = _calc_indicadores(reg)
        linhas.append({
            "Ticker": t,
            "Ano": ano,
            "Liquidez Corrente": inds["LiquidezCorrente"],
            "Liquidez Seca": inds["LiquidezSeca"],
            "Giro do Ativo": inds["GiroAtivo"],
            "PMP (dias)": inds["PMP_dias"]
        })

indicadores_df = pd.DataFrame(linhas)
# Ordena por Ticker e Ano
try:
    indicadores_df = indicadores_df.sort_values(by=["Ticker","Ano"]).reset_index(drop=True)
except Exception:
    pass

print("\n=== INDICADORES FINANCEIROS (2021–2024) ===")
print(indicadores_df.to_string(index=False, float_format=lambda v: f"{v:.2f}" if pd.notnull(v) else "NA"))

# Salva CSV final
OUT_CSV = "indicadores_2021_2024.csv"
indicadores_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"\n[OK] Tabela salva em: {OUT_CSV}")

# (Opcional) Tabela por empresa (pivot) para visual mais rápido:
try:
    piv = indicadores_df.pivot_table(index=["Ticker","Ano"],
                                     values=["Liquidez Corrente","Liquidez Seca","Giro do Ativo","PMP (dias)"],
                                     aggfunc="first")
    print("\n=== VISÃO PIVOT ===")
    print(piv)
except Exception:
    pass

# ============ GRÁFICOS OPCIONAIS ============
# 1) Preço normalizado MENSAL
plt.figure(figsize=(10,5))
(precos_m[ativos_cols] / precos_m[ativos_cols].iloc[0]).plot(ax=plt.gca())
plt.title('Preço Normalizado (Mensal, t0 = 1.0)')
plt.xlabel('Mês'); plt.ylabel('Índice (x vezes)')
plt.legend(title='Ativos'); plt.tight_layout(); plt.show()

# 2) Retorno acumulado (ativos)
ret_acum_df = pd.DataFrame(
    {t: [retorno_acumulado(ret_mensal[t][:i+1]) for i in range(len(ret_mensal[t]))] for t in ativos_cols},
    index=precos_m.index[1:]  # alinha com as datas dos retornos mensais
)
ret_acum_pct = ret_acum_df * 100
plt.figure(figsize=(10,5))
ret_acum_pct.plot(ax=plt.gca())
plt.title('Retorno Acumulado (%) — Mensal (∏(1+R_m) - 1)')
plt.xlabel('Mês'); plt.ylabel('Retorno acumulado (%)')
plt.legend(title='Ativos'); plt.tight_layout(); plt.show()

# 3) Barras de Beta
plt.figure(figsize=(8,4))
pd.Series(betas).sort_values().plot(kind='bar', ax=plt.gca())
plt.title('Betas dos Ativos (referência: Ibovespa)')
plt.ylabel('β'); plt.tight_layout(); plt.show()

# ====================== GRÁFICOS DOS INDICADORES (2021–2024) ======================
import os

# garante que temos o CSV calculado na etapa anterior
INDIC_CSV = "indicadores_2021_2024.csv"
if not os.path.exists(INDIC_CSV):
    print(f"[Aviso] Arquivo {INDIC_CSV} não encontrado. Gere os indicadores antes desta etapa.")
else:
    df_ind = pd.read_csv(INDIC_CSV)

    # limpa/ordena
    df_ind = df_ind.dropna(subset=["Ano"]).copy()
    df_ind["Ano"] = df_ind["Ano"].astype(int)
    df_ind = df_ind.sort_values(["Ticker", "Ano"])

    # pasta de saída
    outdir = "graficos"
    os.makedirs(outdir, exist_ok=True)

    indicadores = ["Liquidez Corrente", "Liquidez Seca", "Giro do Ativo", "PMP (dias)"]

    # 1) Séries temporais por indicador (linhas por ticker)
    for ind in indicadores:
        try:
            piv = df_ind.pivot(index="Ano", columns="Ticker", values=ind).sort_index()
            ax = piv.plot(marker="o", figsize=(9, 5))
            ax.set_title(f"{ind} — (2021–2024)")
            ax.set_xlabel("Ano")
            ax.set_ylabel(ind)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(title="Ticker", frameon=False)
            plt.tight_layout()
            fname = os.path.join(outdir, f"{ind.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}_serie.png")
            plt.savefig(fname, dpi=160)
            plt.show()
            print(f"[OK] Gráfico salvo: {fname}")
        except Exception as e:
            print(f"[Aviso] Não foi possível plotar '{ind}': {e}")

    # 2) Comparativo do ÚLTIMO ANO disponível (barras por ticker)
    try:
        ultimo_ano = int(df_ind["Ano"].max())
        base_ultimo = df_ind[df_ind["Ano"] == ultimo_ano].copy()

        # barras lado a lado: 1 figura por indicador
        for ind in indicadores:
            sub = base_ultimo[["Ticker", ind]].set_index("Ticker").sort_index()
            ax = sub.plot(kind="bar", figsize=(9, 5))
            ax.set_title(f"{ind} — Comparativo ({ultimo_ano})")
            ax.set_xlabel("Ticker")
            ax.set_ylabel(ind)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend().remove()
            plt.tight_layout()
            fname = os.path.join(outdir, f"{ind.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}_comparativo_{ultimo_ano}.png")
            plt.savefig(fname, dpi=160)
            plt.show()
            print(f"[OK] Gráfico salvo: {fname}")
    except Exception as e:
        print(f"[Aviso] Não foi possível gerar comparativos do último ano: {e}")



