# Retail Shelf Space Optimization (Otimização de Gôndola)

Este projeto implementa uma solução de ponta a ponta combinando **Machine Learning (Previsão de Demanda)** e **Pesquisa Operacional (Otimização Combinatória)** para gerenciar o espaço de prateleiras no varejo. 

O objetivo central é maximizar o desempenho de vendas respeitando rigorosamente as regras de governança de Trade Marketing (compliance de categorias e lançamentos).

## 🚀 Visão Geral da Solução

A arquitetura do projeto foi desenhada para separar claramente a inteligência analítica da camada de interface, garantindo robustez e prevenindo "alucinações" numéricas (comuns ao delegar cálculos matemáticos para LLMs).

1.  **Módulo Preditivo (Forecasting):** Modelos baseados em árvore (LightGBM e XGBoost) preveem a demanda futura a nível SKU-Loja com base no comportamento histórico, *lags*, médias móveis e calendário promocional.
2.  **Módulo de Otimização (PuLP):** Utiliza o Algoritmo da Mochila (Knapsack Problem) para determinar o sortimento ideal na prateleira física. Ele confronta as previsões de demanda com as dimensões espaciais (cm) dos produtos e restrições de negócio (mínimo de categorias expostas).
3.  **Camada de Serviço:** Um pipeline *batch* para processamento noturno de relatórios gerenciais e uma API (FastAPI) para consultas *on-demand*.

## 🧠 Principais Decisões Arquiteturais e Científicas

*   **Governança vs. Algoritmo Guloso:** O otimizador linear foi comparado a heurísticas tradicionais. Enquanto o modelo humano atinge falsos picos de vendas ao lotar as gôndolas com itens pequenos e ignorar contratos de exposição visual, o otimizador garante **100% de uso do espaço** e **100% de compliance**, sacrificando um percentual irrelevante de volume em prol da estratégia de marca de longo prazo.
*   **Prevenção de *Data Leakage*:** Como a base possui histórico curto (poucos meses), o uso de *TimeSeriesSplit* tradicional é inviável. Foi adotado o **PredefinedSplit**, que isola rigidamente o último ciclo como validação, impedindo que o modelo treine com dados do futuro (data leakage).
*   **Otimização Bayesiana e *Overfitting*:** Os hiperparâmetros não foram escolhidos aleatoriamente. Utilizou-se o **Optuna** associado à geração de **Validation Curves**, garantindo que as árvores de decisão parassem de crescer no exato momento antes de começarem a decorar o ruído (limite do `max_depth`).
*   **Interpretabilidade (Feature Importance via Gain):** Para lidar com as variáveis binárias de alta relevância (como "Campanhas" e "Lançamentos"), o cálculo de importância no LightGBM foi migrado da régua de 'Split' (frequência) para a métrica de 'Gain' (Ganho de Informação), refletindo perfeitamente as evidências da Análise Exploratória (EDA).

## 📂 Estrutura do Repositório

```text
├── data/
│   ├── raw_data/            # Dados brutos
│   ├── processed_data/      # Dados após Feature Engineering (features.py)
│   ├── predictions/         # Outputs de previsão prontos para o PuLP
│   └── validation/          # CSVs com resultados contra a base de validação
├── docs/
│   └── log_intervencoes.md  # Log detalhado das decisões técnicas
├── notebooks/
│   ├──  01_EDA.ipynb          # Notebook com Analise exploratoria de dados
│   └──  eda_helper.py         # Funcoes para analise exploratoria
├── src/
│   ├── 01_data_prep.py         # Carregamento da tabela Inicial, merge e limpeza
│   ├── 02_features.py          # Engenharia de atributos (lags, rollings)
│   ├── 03_train.py             # Modelo Baseline (XGBoost vs LightGBM)
│   ├── 04_train_turbo.py       # Modelo Final com Optuna e Validation Curves
│   ├── 05_evaluation.py        # Geração de métricas de negócio (MAE, WMAPE, WRMSSE)
│   ├── 06_optimization.py      # Aplicacao do Algoritmo de Otimizacao sobre as previsoes finais
│   ├── 07_baseline_greedy.py   # Baseline de Otimização (Simulação de decisão humana)
│   ├── 08_batch_report_generator.py # Pipeline de geração de relatórios (exemplo - chamada LLM comentada)
│   ├── 09_api_service.py    # Aplicação FastAPI - Exemplo usando apenas dados de otimizacao sem chamada de LLM
│   └── requirements.txt     # Arquivo de requirements
└── README.md
```

## ⚙️ Como Executar

### 1. Preparando o Ambiente
Recomenda-se o uso de um ambiente virtual (`.venv`).
```bash
pip install -r requirements.txt
# ou instale manualmente:
pip install pandas numpy scikit-learn lightgbm xgboost optuna pulp fastapi uvicorn matplotlib seaborn
```

### 2. Pipeline de Machine Learning
Execute os scripts na ordem conforme o nr para treinar, até o evalation (nrs 01 a 05). Exemplo:
```bash
python src/01_data_prep.py       

```

### 3. Pipeline de Otimização
Execute os scripts na ordem , nrs 06 e 07. Exemplo:

```bash
python src/06_optimization.py       

```

### 4. Executando testes para Aplicação para servir as franquias (mock up)
Scripts nrs 08 e 09
```bash
python src/08_batch_report_generator.py       

```
Para conferir se a api subiu (status 200):

http://0.0.0.0:8000/

E a documentacao no swagger para consulta:

http://127.0.0.1:8000/docs#/