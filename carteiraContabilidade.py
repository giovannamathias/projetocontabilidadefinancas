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
