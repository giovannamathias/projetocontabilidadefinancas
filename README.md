# 📊 Projeto – Cálculo de Indicadores Financeiros e CAPM  
**Universidade de Pernambuco – UPE Caruaru**  
**Disciplina:** Fundamentos de Contabilidade e Finanças  

---

## 🧠 Descrição do Projeto  

Este projeto integra **análise de investimentos e indicadores contábeis**, permitindo estudar o **desempenho financeiro de empresas brasileiras** com base em dados reais extraídos do Yahoo Finance.  

O script realiza:  
- 📈 Coleta automática de preços (ações e Ibovespa);  
- 🧮 Cálculo de retornos, volatilidade, correlação, Betas e CAPM;  
- 🏦 Extração de indicadores contábeis (2021–2024) de forma manual ou automática;  
- 📊 Geração de gráficos comparativos e séries históricas dos principais índices de liquidez, eficiência e solvência;  
- 📑 Exportação automática dos resultados para arquivos CSV e PNG.  

---

## ⚙️ Funcionalidades Principais  

### 1️⃣ Análise de Mercado e Risco  
Coleta cotações de **PETR4.SA**, **VALE3.SA** e **ITUB4.SA**.  

Calcula:  
- Retorno médio e desvio padrão (mensal e anual);  
- Beta e CAPM (mensal e anual);  
- Correlações entre os ativos;  
- Risco e retorno esperado da carteira.  

---

### 2️⃣ Indicadores Contábeis (2021–2024)  
- Liquidez Corrente  
- Liquidez Seca  
- Giro do Ativo  
- Prazo Médio de Pagamento (PMP)  

---

### 3️⃣ Indicadores Extras de Estrutura e Solvência  
*(Baseados nos dados do Balanço Patrimonial e DRE preenchidos em `indicadores_input.csv`)*  

- Liquidez Imediata  
- Liquidez Geral  
- Solvência  
- Endividamento Total  
- Capital Circulante Líquido (CCL)  

---

### 4️⃣ Geração de Gráficos Automáticos  
São criados arquivos PNG em `/graficos`, como:  

- `Giro_do_Ativo_serie.png`  
- `Liquidez_Seca_comparativo_2024.png`  
- `Solvência_serie_extra.png`  
- `Endividamento_Total_comparativo_2024_extra.png`  

---

## 🧩 Tecnologias Utilizadas  

- 🐍 Python 3.11+  
- 📊 pandas  
- 📈 matplotlib  
- 💸 yfinance  



