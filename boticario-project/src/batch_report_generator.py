import os
import pandas as pd
from pydantic import BaseModel, Field

class RelatorioFranqueado(BaseModel):
    saudacao_personalizada: str = Field(description="Boas vindas calorosas citando o formato (tp_perfil) e regiao da loja.")
    analise_estrategica: str = Field(description="1 paragrafo explicando por que os top 3 produtos e lancamentos enviados sao a aposta do ciclo.")
    alerta_operacional: str = Field(description="1 paragrafo curto com dicas de exposicao na gondola e atencao ao estoque para evitar ruptura.")

class BatchReportGenerator:
    def __init__(self, gondola_path="../data/predictions/gondola_202609.csv", previsao_path="../data/predictions/previsao_202609_full.csv"):
        self.gondola_path = gondola_path
        self.previsao_path = previsao_path
        self.output_dir = "../data/reports/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6).with_structured_output(RelatorioFranqueado)
        
        self.prompt_template = '''Você é um consultor estratégico de varejo d'O Boticário.
Sua tarefa é redigir a análise qualitativa para o franqueado de uma loja com base nos KPIs matemáticos abaixo.
A saída deve ser OBRIGATORIAMENTE no formato JSON solicitado pela ferramenta.

DADOS DA LOJA {loja_id}:
- Perfil: {perfil}
- Região: {regiao}
- Receita Potencial Projetada do Ciclo: R$ {faturamento_projetado:.2f}
- Top 3 Produtos em Volume: {top_3_skus}
- Lançamentos Recomendados na Gôndola: {lancamentos}'''

    def load_data(self):
        print("Carregando bases otimizadas para geração dos relatórios...")
        df_full = pd.read_csv(self.previsao_path)
        df_gondola = pd.read_csv(self.gondola_path)
        self.df_final = df_gondola.merge(df_full, on=['cod_loja', 'cod_sku'], how='left')

    def generate_batch(self, max_lojas=3):
        print("\n=== Iniciando Processamento Batch (LLM) ===")
        lojas = self.df_final['cod_loja'].unique()
        
        if max_lojas:
            lojas = lojas[:max_lojas]
            print(f"Modo de Teste: Processando apenas {max_lojas} lojas.")
            
        for loja in lojas:
            print(f"Gerando relatório para a loja: {loja}...")
            df_loja = self.df_final[self.df_final['cod_loja'] == loja].copy()
            
            regiao = df_loja['des_regiao'].iloc[0]
            perfil = df_loja['tp_perfil'].iloc[0]
            
            df_loja['receita'] = df_loja['qt_venda_prevista'] * df_loja['vlr_preco_tabela']
            faturamento_projetado = df_loja['receita'].sum()
            
            top_3 = df_loja.nlargest(3, 'qt_venda_prevista')['cod_sku'].tolist()
            lancamentos = df_loja[df_loja['flg_lancamento'] == 1]['cod_sku'].tolist()
            
            class MockResponse:
                saudacao_personalizada = f"Olá, equipe da loja {loja}! Sabemos que o formato {perfil} na região {regiao} tem um ritmo acelerado."
                analise_estrategica = f"Para este ciclo, miramos num potencial de R$ {faturamento_projetado:.2f}. A inteligência de dados priorizou {', '.join(top_3)}."
                alerta_operacional = "Garanta que as vitrines principais destaquem esses itens para não perdermos nenhuma conversão por ruptura."
            
            resposta_ia = MockResponse()

            relatorio_md = f"# Estratégia de Gôndola - Ciclo 202609 | Loja: {loja}\n\n"
            relatorio_md += f"## Mensagem do Consultor IA\n"
            relatorio_md += f"{resposta_ia.saudacao_personalizada}\n\n"
            relatorio_md += f"{resposta_ia.analise_estrategica}\n\n"
            relatorio_md += f"**Ação Operacional:** {resposta_ia.alerta_operacional}\n\n"
            
            relatorio_md += f"## Seu Planograma Otimizado (Gôndola)\n"
            relatorio_md += df_loja[['cod_sku', 'des_categoria', 'qt_venda_prevista', 'qt_espaco']].to_markdown(index=False)
            
            file_path = os.path.join(self.output_dir, f"relatorio_{loja}.md")
            with open(file_path, "w", encoding='utf-8') as f:
                f.write(relatorio_md)
                
        print(f"\nSucesso! Relatórios gerados em: {self.output_dir}")

if __name__ == "__main__":
    gerador = BatchReportGenerator()
    gerador.load_data()
    gerador.generate_batch(max_lojas=3)
