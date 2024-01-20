from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer

# Инициализация SparkSession
spark = SparkSession.builder.appName("NewsWord2Vec").getOrCreate()

# Загрузка данных из JSON-файлов в DataFrame
news_df = spark.read.json("D:/news_json/*.json")

# Токенизация текста
tokenizer = Tokenizer(inputCol="text", outputCol="words")
news_df = tokenizer.transform(news_df)

# Обучение модели word2vec
word2Vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="wordvectors")
model = word2Vec.fit(news_df)
result = model.transform(news_df)

# Функция для поиска контекстных синонимов
def find_synonyms(word, num_synonyms=5):
    synonyms = model.findSynonyms(word, num_synonyms)
    return [row.word for row in synonyms.collect()]

# Пример использования
input_word = "синоним"
synonyms = find_synonyms(input_word)
print(f"Контекстные синонимы для слова '{input_word}': {synonyms}")

# Завершение сессии Spark
spark.stop()