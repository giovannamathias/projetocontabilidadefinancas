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
HOJE = datetime.today().date()
START = (HOJE - timedelta(days=365*4 + 20)).isoformat()
END = HOJE.isoformat()
ORCAMENTO_TOTAL = 100000.00
MESES_ANO = 12

# ============ DOWNLOAD (Adj Close) ============
print(f"Baixando dados de {TICKERS} entre {START} e {END}...")
precos_d = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)['Close']
if isinstance(precos_d, pd.Series):
    precos_d = precos_d.to_frame(name=TICKERS[0])

precos_d = precos_d.dropna(how='all').sort_index()
precos_d = precos_d.dropna(how='any')             # exige cotação de todos no dia

# AGREGA MENSAL: último preço ajustado de cada mês
precos_m = precos_d.resample('M').last().dropna(how='any')

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

# ============ RETORNOS MENSAIS & ESTATÍSTICAS ============
ret_mensal = {}      # ticker -> lista de retornos mensais
estat = {}           # métricas por ativo

for t in TICKERS:
    serie = precos_m[t].tolist()
    r = retornos_mensais(serie)
    ret_mensal[t] = r

    mu_m = media(r)
    sd_m = desvio_padrao(r)

    # anualização a partir de mensais
    mu_a = mu_m * MESES_ANO
    sd_a = sd_m * math.sqrt(MESES_ANO)
    cv_a = (sd_a / mu_a) if mu_a != 0 else float('nan')
    ret_acum_4a = retorno_acumulado(r)

    estat[t] = {
        'PrecoAtual': serie[-1],
        'RetornoMedio_m': mu_m,
        'DesvioPadrao_m': sd_m,
        'RetornoMedio_a': mu_a,
        'DesvioPadrao_a': sd_a,
        'CV_anual': cv_a,
        'RetAcum_4a': ret_acum_4a
    }

resumo_ativos = pd.DataFrame(estat).T

# ============ MATRIZ DE CORRELAÇÃO (mensal) ============
N = len(TICKERS)
corr_matrix = pd.DataFrame(index=TICKERS, columns=TICKERS, dtype=float)
for i in range(N):
    for j in range(N):
        x = ret_mensal[TICKERS[i]]
        y = ret_mensal[TICKERS[j]]
        corr_matrix.iloc[i, j] = 1.0 if i == j else correlacao(x, y)

print("\n=== MATRIZ DE CORRELAÇÃO (mensal) ===")
print(corr_matrix.round(4))

# ---- Correlação 2 a 2 (lista de pares)
print("\n=== CORRELAÇÃO 2 a 2 (mensal) ===")
for a, b in combinations(TICKERS, 2):
    rho = correlacao(ret_mensal[a], ret_mensal[b])
    print(f"{a}  x  {b}:  {rho:.3f}")

# ============ DETALHES POR AÇÃO ============
print("\n=== DETALHES POR AÇÃO (base MENSAL → anualizada) ===")
for t in TICKERS:
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

# ============ RANKINGS (opcionais pro relatório) ============
print("\n=== RANKING — Retorno médio anual (maior → menor) ===")
print(resumo_ativos['RetornoMedio_a'].sort_values(ascending=False).apply(lambda v: f"{v*100:.2f}%"))

print("\n=== RANKING — CV anual (menor → melhor) ===")
print(resumo_ativos['CV_anual'].sort_values().apply(lambda v: f"{v:.3f}x"))

# ============ CARTEIRA (pesos iguais) — métricas ANUAIS ============
w = [1 / N] * N
# retorno esperado anual: média ponderada das médias anuais
mu_port_a = sum(w[i] * estat[TICKERS[i]]['RetornoMedio_a'] for i in range(N))

# variância anual: Σ_i Σ_j w_i w_j * (cov_mensal * 12)
sig2_port_a = 0.0
for i in range(N):
    for j in range(N):
        cov_ij_m = covariancia(ret_mensal[TICKERS[i]], ret_mensal[TICKERS[j]])
        cov_ij_a = cov_ij_m * MESES_ANO
        sig2_port_a += w[i] * w[j] * cov_ij_a
sigma_port_a = math.sqrt(sig2_port_a)

# ============ ALOCAÇÃO REAL ============
ultimos_precos = precos_m.iloc[-1]
qtd = ((ORCAMENTO_TOTAL / N) // ultimos_precos).astype(int)  # inteiras
investido = qtd * ultimos_precos
total_invest = float(investido.sum())
pesos_reais = (investido / total_invest)

# retorno esperado anual com pesos reais
mu_port_real_a = sum(pesos_reais[i] * estat[TICKERS[i]]['RetornoMedio_a'] for i in range(N))

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

# ============ GRÁFICOS ============
# 1) Preço normalizado MENSAL
plt.figure(figsize=(10,5))
(precos_m / precos_m.iloc[0]).plot(ax=plt.gca())
plt.title('Preço Normalizado (Mensal, t0 = 1.0)')
plt.xlabel('Mês'); plt.ylabel('Índice (x vezes)')
plt.legend(title='Ativos'); plt.tight_layout(); plt.show()

# 2) Retorno acumulado
ret_acum_df = pd.DataFrame(
    {t: [retorno_acumulado(ret_mensal[t][:i+1]) for i in range(len(ret_mensal[t]))] for t in TICKERS},
    index=precos_m.index[1:]  # alinha com as datas dos retornos mensais
)
ret_acum_pct = ret_acum_df * 100
plt.figure(figsize=(10,5))
ret_acum_pct.plot(ax=plt.gca())
plt.title('Retorno Acumulado (%) — Mensal (∏(1+R_m) - 1)')
plt.xlabel('Mês'); plt.ylabel('Retorno acumulado (%)')
plt.legend(title='Ativos'); plt.tight_layout(); plt.show()
