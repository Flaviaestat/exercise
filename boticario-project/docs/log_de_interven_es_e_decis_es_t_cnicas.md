# Log de Intervenções e Decisões de Arquitetura

Este documento registra o histórico de refinamentos técnicos, discussões de negócio e correções de código realizados durante o desenvolvimento do pipeline de Machine Learning e Otimização Matemática.
A LLM usada para acelerar o desenvolvimento do projeto foi o Gemini Pro com alto esforço. No entanto várias análises e intervencoes foram necessárias. Optou-se por um AI menos "eager" para se manter o controle e narrativa das decisoes.

## 1. Estratégia de Arquitetura e Governança
*   **Decisão:** Adoção de uma arquitetura híbrida (Batch para processamento noturno via `batch_report_generator.py` e API *on-demand* via FastAPI).
*   **Mitigação de Alucinação:** Implementação do conceito de *Layered Generation*. Toda a matemática de otimização é executada deterministicamente no Pandas/PuLP, enquanto o LLM é restrito exclusivamente à geração de texto explicativo no front-end, garantindo 100% de precisão nos números finais.

## 2. Algoritmo Guloso vs. Otimização Linear (PuLP)
*   **Discussão:** Análise do paradoxo onde o algoritmo guloso projetava mais vendas físicas do que o otimizador matemático.
*   **Intervenção:** Ajuste da narrativa de negócio. Identificou-se que o algoritmo guloso "trapaceava" preenchendo a gôndola apenas com itens pequenos e de altíssimo giro (ex: maquiagem), violando regras de sortimento.
*   **Conclusão:** O PuLP sacrificou um volume mínimo de unidades (cerca de 0.3%) para conseguir alocar itens maiores obrigatórios (ex: corpo/perfumaria), atingindo 100% de *compliance* com o Trade Marketing e mantendo 100% de ocupação física. O foco mudou de "Uplift de Vendas" para "Governança e Compliance".

## 3. Prevenção de Data Leakage em Séries Temporais Curtas
*   **Problema:** Histórico de dados muito curto (apenas 4 ciclos), o que inviabilizava métodos tradicionais como `TimeSeriesSplit`.
*   **Intervenção:** Implementação do `PredefinedSplit` do `scikit-learn` no script de treino turbo. Isso permitiu "congelar" o ciclo `202608` estritamente como validação, utilizando ciclos anteriores como treino, bloqueando vazamento de dados (data leakage) e simulando a vida real.

## 4. Otimização Bayesiana e Overfitting
*   **Intervenção:** Criação do script `train_turbo.py` utilizando **Optuna** para busca inteligente de hiperparâmetros (substituindo o GridSearchCV).
*   **Evidência Visual:** Adição do método para gerar a **Validation Curve** (focada no `max_depth`). O objetivo foi provar para a banca de avaliação o momento exato em que a árvore começa a decorar os dados (overfitting), justificando o corte da profundidade para manter a generalização do modelo.

## 5. Correção de Features (Data Types e Vieses)
*   **Problema (Dtypes):** Erro de `ValueError` no LightGBM ao tentar ler colunas do tipo `object`. 
*   **Intervenção:** Conversão forçada de variáveis categóricas explicativas (`tp_perfil`, `des_regiao`, `des_categoria`) para `.astype('category')`.
*   **Problema (Vazamento de Identificadores):** O modelo base estava apresentando resultados irreais de treino porque incluía `cod_loja` e `cod_sku` nas features, o que permitia ao modelo "decorar" o comportamento ao invés de aprender padrões temporais.
*   **Intervenção:** Remoção estrita de IDs da lista de features preditivas tanto no `train.py` quanto no `train_turbo.py`, forçando as árvores a aprenderem através de *lags*, médias móveis e indicadores de campanhas.

## 6. O Paradoxo do Feature Importance
*   **Problema:** A Análise Exploratória (EDA) indicava que "Campanha" era vital para as vendas, mas o gráfico inicial de *Feature Importance* do LightGBM a mostrava como irrelevante.
*   **Investigação:** Identificou-se que o LightGBM estava utilizando a régua padrão de `split` (frequência de cortes), que penaliza variáveis binárias (como flags de promoção) em relação a variáveis contínuas (como preço).
*   **Intervenção:** Alteração do código de extração de métricas no `train.py` para `importance_type='gain'`. Isso alinhou o racional matemático do LightGBM com o do XGBoost, resgatando o peso preditivo correto da variável de campanha e validando as conclusões da EDA.

## 7. Scripts Auxiliares Gerados
*   `plot_previsto_real.py`: Desenvolvido para gerar gráficos de dispersão comparando a aderência (linha de 45 graus) entre Vendas Reais vs. Vendas Previstas usando o output de validação.