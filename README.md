
# Análise e Classificação de Promoções para Otimização de Consumo 

Este projeto completo utiliza PySpark para processar, limpar e modelar dados relacionados a interações de clientes com promoções. O objetivo final é criar um modelo de Machine Learning capaz de prever quais promoções têm maior probabilidade de gerar alto consumo, permitindo que as empresas otimizem suas estratégias de marketing e melhorem o engajamento do cliente.

## Visão Geral
No dinâmico cenário do varejo e serviços, a personalização de ofertas é um diferencial competitivo. Este projeto aborda a necessidade de identificar e prever o sucesso de campanhas promocionais. Ao integrar dados de transações, perfis de clientes e detalhes de ofertas, construímos um pipeline de dados robusto e um modelo de classificação. Este modelo nos ajuda a entender não apenas quais ofertas são bem-sucedidas, mas também porque, capacitando a tomada de decisões baseada em dados para futuras campanhas.

## Estrutura do Projeto
O projeto é organizado em etapas sequenciais, cada uma com um objetivo claro:

### 1_data_processing.ipynb (Ingestão de Dados):
Baixa um arquivo .tar.gz contendo dados brutos de uma URL remota.
Extrai os arquivos JSON (offers.json, profile.json, transactions.json) para o diretório data/raw.
Carrega esses JSONs em DataFrames PySpark para inspeção inicial.
Limpeza e Padronização de Dados (offers_cleaned, customers_cleaned, transactions_cleaned):
Scripts específicos para cada DataFrame (df_offers, df_customers, df_transactions).
Realizam validações, tratamento de nulos, padronização de formatos e tipos de dados.
Salvam os DataFrames limpos no diretório data/processed.

### 2_modeling.ipynb (Construção e Treinamento do Modelo):
Lê os DataFrames limpos do diretório data/processed.
Realiza engenharia de features combinando os dados de transações, clientes e ofertas.
Cria uma label (rótulo) binária para indicar o sucesso da promoção com base no consumo.
Utiliza um Pipeline PySpark ML com StringIndexer (para categóricos), VectorAssembler (para unificar features) e GBTClassifier (modelo de classificação).
Divide os dados em conjuntos de treinamento e teste.
Treina o modelo e avalia seu desempenho usando AUC.

## Pré-requisitos
Para executar este projeto, você precisará ter o seguinte instalado e configurado:

Python 3.x
Apache Spark (versão 3.x recomendada)
PySpark
Java Development Kit (JDK) (compatível com sua versão do Spark)

## Como Executar
Siga os passos abaixo na ordem para executar o projeto completo:

Processamento de dados:
Execute o script de ingestão para baixar e extrair os dados brutos.

Treinamento do Modelo:
Execute o script de treinamento do modelo, que lerá os dados limpos, fará a engenharia de features e treinará o GBTClassifier.

## Autor
Projeto desenvolvido por Maíra Duran Baldissera, com foco em aplicações práticas de Machine Learning com Big Data para marketing personalizado.
