# Retail Shelf Space Optimization (Otimização de Gôndola)

Este projeto implementa uma solução de ponta a ponta (End-to-End) combinando **Machine Learning (Previsão de Demanda)** e **Pesquisa Operacional (Otimização Combinatória)** para gerenciar o espaço de prateleiras no varejo. 

O objetivo central é maximizar o desempenho de vendas respeitando rigorosamente as regras de governança de Trade Marketing (compliance de categorias e lançamentos).

## 🚀 Visão Geral da Solução

A arquitetura do projeto foi desenhada para separar claramente a inteligência analítica da camada de interface, garantindo robustez e prevenindo "alucinações" numéricas (comuns ao delegar cálculos matemáticos para LLMs).

1.  **Módulo Preditivo (Forecasting):** Modelos baseados em árvore (LightGBM e XGBoost) preveem a demanda futura a nível SKU-Loja com base no comportamento histórico, *lags*, médias móveis e calendário promocional.
2.  **Módulo de Otimização (PuLP):** Utiliza o Algoritmo da Mochila (Knapsack Problem) para determinar o sortimento ideal na prateleira física. Ele confronta as previsões de demanda com as dimensões espaciais (cm) dos produtos e restrições de negócio (mínimo de categorias expostas).
3.  **Camada de Serviço:** Um pipeline *batch* para processamento noturno de relatórios gerenciais e uma API (FastAPI) para consultas *on-demand*.

## 🧠 Principais Decisões Arquiteturais e Científicas

*   **Governança vs. Algoritmo Guloso:** O otimizador linear provou superar heurísticas tradicionais (métodos gulosos). Enquanto o modelo humano atinge falsos picos de vendas ao lotar as gôndolas com itens pequenos e ignorar contratos de exposição visual, o otimizador garante **100% de uso do espaço** e **100% de compliance**, sacrificando um percentual irrelevante de volume em prol da estratégia de marca de longo prazo.
*   **Prevenção de *Data Leakage*:** Como a base possui histórico curto (poucos meses), o uso de *TimeSeriesSplit* tradicional é inviável. Foi adotado o **PredefinedSplit**, que isola rigidamente o último ciclo como validação, impedindo que o modelo treine com dados do futuro.
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
│   ├── images/              # Gráficos de feature importance, validation curves, etc.
│   └── log_intervencoes.md  # Log detalhado das decisões técnicas
├── src/
│   ├── features.py          # Engenharia de atributos (lags, rollings)
│   ├── train.py             # Modelo Baseline (XGBoost vs LightGBM)
│   ├── train_turbo.py       # Modelo Final com Optuna e Validation Curves
│   ├── evaluation.py        # Geração de métricas de negócio (MAE, WMAPE, WRMSSE)
│   ├── plot_previsto_real.py# Gráficos de validação visual
│   ├── baseline_greedy.py   # Baseline de Otimização (Simulação de decisão humana)
│   ├── batch_report_generator.py # Pipeline de geração de relatórios
│   └── api_service.py       # Aplicação FastAPI
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
Execute os scripts na seguinte ordem para treinar o modelo a partir do zero:
```bash
python src/features.py        # 1. Gera variáveis temporais
python src/train.py           # 2. Executa o modelo de baseline
python src/train_turbo.py     # 3. Executa a Otimização Bayesiana e gera as previsões finais
python src/plot_previsto_real.py # 4. Gera gráficos de validação
```

### 3. Executando a Aplicação de Negócio
Para processar os relatórios em lote:
```bash
python src/batch_report_generator.py
```
Para subir a API de consulta on-demand:
```bash
uvicorn src.api_service:app --reload
```
