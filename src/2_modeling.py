# %%
# Notebook: Classificação de Promoções para Maximizar Consumo de Clientes

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Inicialização
spark = SparkSession.builder.appName("PromotionClassificationModel").getOrCreate()

# Leitura dos dados
df_transactions = spark.read.json("D:\\ProjetosEstudo\\Case_ifood\\CaseIfood\\ifood-case\\data\\processed\\transactions_cleaned/part-*.json")
df_customers = spark.read.json("D:\\ProjetosEstudo\\Case_ifood\\CaseIfood\\ifood-case\\data\\processed\\customers_cleaned/part-*.json")
df_offers = spark.read.json("D:\\ProjetosEstudo\\Case_ifood\\CaseIfood\\ifood-case\\data\\processed\\offers_cleaned/part-*.json")
df_transactions.show()
df_offers.show()
df_customers.show()

# Filtragem de eventos
received = df_transactions.filter(col("event") == "offer received")
completed = df_transactions.filter(col("event") == "offer completed")
transactions = df_transactions.filter(col("event") == "transaction")


# União de dataframes para recuperar o tempo de vigência de uma promoção
completed = completed.select("account_id", "offer_id", "time_since_test_start")
completed = completed.withColumnRenamed("time_since_test_start", "time_end")
time_offers = received.join(completed, on=["account_id","offer_id"], how="inner")

# Adiciona data de início da oferta ao cliente
offers_sent = time_offers.withColumn("start", col("time_since_test_start")) \
                       .withColumn("end", col("time_end"))

# Associa transações que ocorreram durante a vigência da oferta para aquele cliente
transacoes_com_oferta = transactions.alias("t") \
    .join(offers_sent.alias("o"),
          (col("t.account_id") == col("o.account_id")) &
          (col("t.time_since_test_start") >= col("o.start")) &
          (col("t.time_since_test_start") <= col("o.end")),
          "inner") \
    .select(col("t.account_id"), col("o.offer_id"), col("t.amount"))

# Soma total consumida por cliente+oferta (proxy do sucesso da promo)
consumo = transacoes_com_oferta.groupBy("account_id", "offer_id") \
                     .agg(spark_sum("amount").alias("total_consumo"))

# 8. Construção do dataset supervisionado: received + consumo + clientes + ofertas
base = received.join(consumo, on=["account_id", "offer_id"], how="left") \
               .join(df_customers, received.account_id == df_customers.customer_id, "left") \
               .join(df_offers, received.offer_id == df_offers.id, "left")

# Preenche consumo nulo com 0
base = base.withColumn("total_consumo", when(col("total_consumo").isNull(), 0).otherwise(col("total_consumo")))

# Cria label binária: 1 se consumo total é alto o suficiente (ex: > 10)
base = base.withColumn("label", when(col("total_consumo") >= 10, 1).otherwise(0))

# Indexação de variáveis categóricas
indexers = [
    StringIndexer(inputCol="gender", outputCol="gender_index"),
    StringIndexer(inputCol="offer_type", outputCol="offer_type_index")
]

# Vetorizador
assembler = VectorAssembler(
    inputCols=["age", "credit_card_limit", "discount_value", "duration", "min_value", "reward", "gender_index", "offer_type_index"],
    outputCol="features"
)

# Modelo
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt])

# Treinamento e teste
train, test = base.randomSplit([0.8, 0.2], seed=42)
train.show()
train = train.filter(col("customer_id").isNotNull() & col("id").isNotNull())
test = test.filter(col("customer_id").isNotNull() & col("id").isNotNull())
model = pipeline.fit(train)

# Avaliação
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")