# %% [markdown]
# Carregando dados

# %%
import os
import tarfile
import requests
from pyspark.sql import SparkSession

# Inicializa sessão Spark
spark = SparkSession.builder.getOrCreate()

# Define diretório target e URL
url = "https://data-architect-test-source.s3.sa-east-1.amazonaws.com/ds-technical-evaluation-data.tar.gz"
raw_dir = os.path.join(os.getcwd().split('\\notebooks')[0], "data\\raw")
processed_dir = os.path.join(os.getcwd().split('\\notebooks')[0], "data\\processed")
os.makedirs(raw_dir, exist_ok=True)

# Carrega o arquivo .tar.gz 
local_tar_path = os.path.join(raw_dir, "data_archive.tar.gz")

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(local_tar_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Carregado para: {local_tar_path}")
else:
    raise Exception(f"Download falhou com código de status: {response.status_code}")

# Step 3: Extrai na pasta raw
try:
    with tarfile.open(local_tar_path, mode="r:*") as tar:
        tar.extractall(path=raw_dir)
    print(f"Arquivos extraídos para: {raw_dir}")
except tarfile.ReadError as e:
    raise Exception("Extração falhou: não é um arquivo tar válido") from e

# Step 4: Lista arquivos extraídos
print("Arquivos extraídos:")
for root, dirs, files in os.walk(raw_dir):
    for name in files:
        print(os.path.join(root, name))

# Step 4: Dynamically find the JSON file paths
offer_path = ''
customer_path = ''
transaction_path = ''

lista_nome_arquivos = ['ds-technical-evaluation-data\\offers.json', 
                        'ds-technical-evaluation-data\\profile.json', 
                        'ds-technical-evaluation-data\\transactions.json']
extract_path = raw_dir  

for nome in lista_nome_arquivos:
    if nome == 'ds-technical-evaluation-data\\offers.json':
        offer_path = os.path.join(extract_path, nome)
    elif nome == 'ds-technical-evaluation-data\\profile.json':
        customer_path = os.path.join(extract_path, nome)
    elif nome == 'ds-technical-evaluation-data\\transactions.json':
        transaction_path = os.path.join(extract_path, nome)

# Ensure all files were found


print(f'OFFERPATH:{offer_path}')
print(f'CUSTOMERPATH:{customer_path}')
print(f'TRANSACTIONPATH:{transaction_path}')
#raise FileNotFoundError("Um ou mais arquivos JSON não foram encontrados após extração.")

try:
    # whatever block of code may be failing
    df_offers = spark.read.option("multiline", True).json(f"file:///{offer_path}")
except Exception as e:
    print("Original error:", str(e))

try:
    # whatever block of code may be failing
    df_customers = spark.read.option("multiline", True).json(f"file:///{customer_path}")
except Exception as e:
    print("Original error:", str(e))

try:
    # whatever block of code may be failing
    df_transactions = spark.read.option("multiline", True).json(f"file:///{transaction_path}")
except Exception as e:
    print("Original error:", str(e))


# Step 6: Preview
print("=== OFFERS ===")
df_offers.show(truncate=False)

print("=== CUSTOMERS ===")
df_customers.show(truncate=False)

print("=== TRANSACTIONS ===")
df_transactions.show(truncate=False)









# %% [markdown]
# Filtrando offers e salvando 

# %%
from pyspark.sql.functions import col, size, lower
import os
import shutil
import glob


# Remover registros com ID nulo ou vazio
df_offers= df_offers.filter((col("id").isNotNull()) & (col("id") != ""))

# Verificar e tratar valores inconsistentes nos campos categóricos
valid_types = ["bogo", "discount", "informational"]
df_offers = df_offers.withColumn("offer_type", col("offer_type").cast("string"))
df_offers = df_offers.filter(col("offer_type").isin(valid_types))

# Preencher valores nulos ou ausentes
df_offers = df_offers.fillna({
    "duration": 0,
    "discount_value": 0,
    "min_value": 0
})

# Validar que canais existam (lista não vazia)
df_offers = df_offers.filter((col("channels").isNotNull()) & (size(col("channels")) > 0))

# Ajustar campos de texto para lowercase (padronização)
df_offers = df_offers.withColumn("offer_type", lower(col("offer_type")))

#Salvar o DataFrame tratado
df_offers.coalesce(1).write.mode("overwrite").json(os.path.join(processed_dir, "offers_cleaned"))





# %% [markdown]
# Filtrando customers e salvando

# %%
from pyspark.sql.functions import col, to_date, trim

# Remover registros com campos essenciais nulos
df_customers = df_customers.filter(
    (col("id").isNotNull()) &
    (col("age").isNotNull()) &
    (col("credit_card_limit").isNotNull()) &
    (col("registered_on").isNotNull())
)

# Corrigir e padronizar campo "gender"
# a) Remover espaços extras
df_customers = df_customers.withColumn("gender", trim(col("gender")))

# b) Manter apenas os gêneros válidos
valid_genders = ["M", "F", "O"]
df_customers = df_customers.filter((col("gender").isin(valid_genders)) & (col("gender").isNotNull()))

# Filtrar idades válidas
df_customers = df_customers.filter((col("age") >= 10) & (col("age") <= 100) & (col("age").isNotNull()))

# Converter `registered_on` para tipo data
df_customers = df_customers.withColumn(
    "registered_on", to_date(col("registered_on"), "yyyy-MM-dd")
)

# Renomear `id` para `customer_id` (boa prática para joins)
df_customers = df_customers.withColumnRenamed("id", "customer_id")

# Salvar o DataFrame tratado
df_customers.coalesce(1).write.mode("overwrite").json(os.path.join(processed_dir, "customers_cleaned"))

# %% [markdown]
# Filtrando transactions e salvando

# %%
from pyspark.sql.functions import col, trim, when, struct

# 1. Remover registros sem account_id ou event
df_transactions = df_transactions.filter(
    (col("account_id").isNotNull()) &
    (col("event").isNotNull())
)
# Desmontar a tupla
df_transactions = df_transactions.withColumn("amount", col("value.amount")) \
       .withColumn("offer id", col("value.offer id")) \
       .withColumn("offer_id", col("value.offer_id")) \
       .withColumn("reward", col("value.reward"))
df_trasactions = df_transactions.drop("value")

# Padronizar os eventos em lowercase e sem espaços extras
df_transactions = df_transactions.withColumn("event", trim(col("event")))

# Unir coluna offer id e offer_id para valores não nulos
df_transactions = df_transactions.withColumn(
    "offer_id",
    when(col("offer_id").isNotNull(), col("offer_id"))
    .otherwise(col("offer id"))
)

# Remover a coluna redundante com espaço
df_transactions = df_transactions.drop("offer id")

# Preencher campos numéricos ausentes com 0 (opcional e seguro se reward e amount forem esparsos)
df_transactions = df_transactions.fillna({
    "amount": 0.0,
    "reward": 0.0
})

# Validar tipos dos eventos conhecidos
df_transactions.select("event").distinct().show()
valid_events = ["offer received", "offer viewed", "offer completed", "transaction"]
df_transactions = df_transactions.filter(col("event").isin(valid_events))

# Salvar o DataFrame tratado
df_transactions.coalesce(1).write.mode("overwrite").json(os.path.join(processed_dir, "transactions_cleaned"))