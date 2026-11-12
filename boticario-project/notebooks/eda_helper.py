import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_impact_features(df: pd.DataFrame):
    """
    Gera três visualizações focadas no impacto de negócio:
    1. Vendas por Destaque em Campanha
    2. Vendas ao longo do Tempo (Tendência/Sazonalidade)
    3. Impacto de Lançamentos
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Impacto do Destaque na Campanha (flg_campanha)
    # Apenas para ciclos de campanha (flg_campanha_ciclo == 1)
    df_camp = df[df['flg_campanha_ciclo'] == 1]
    sns.barplot(
        data=df_camp, x='flg_campanha', y='qt_venda', 
        ax=axes[0], errorbar=None, palette='viridis'
    )
    axes[0].set_title('Média de Vendas: Destaque vs Sem Destaque\n(Apenas Ciclos de Campanha)')
    axes[0].set_xlabel('Produto em Destaque (flg_campanha)')
    axes[0].set_ylabel('Quantidade Média Vendida')

    # 2. Tendência ao longo dos ciclos (nr_ordem / cod_ciclo)
    sns.lineplot(
        data=df, x='nr_ordem', y='qt_venda', 
        ax=axes[1], marker='o', errorbar=None, color='b'
    )
    axes[1].set_title('Evolução Média de Vendas por Ciclo')
    axes[1].set_xlabel('Ordem do Ciclo (nr_ordem)')
    axes[1].set_ylabel('Quantidade Média Vendida')

    # 3. Vendas: Lançamento vs Regular (flg_lancamento)
    sns.boxplot(
        data=df, x='flg_lancamento', y='qt_venda', 
        ax=axes[2], showfliers=False, palette='Set2'
    )
    axes[2].set_title('Distribuição de Vendas: Lançamentos vs Linha Regular\n(Sem Outliers)')
    axes[2].set_xlabel('É Lançamento (flg_lancamento)')
    axes[2].set_ylabel('Quantidade Vendida')

    plt.tight_layout()
    plt.show()