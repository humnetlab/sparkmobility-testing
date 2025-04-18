import configparser
import functools
from pyspark.sql import SparkSession
from sparkmobility import config

def create_spark_session():
    print(config)
    spark = (SparkSession
             .builder
             .master(f"local[{config['CORES']}]")
             .appName("SparkMobility")
             .config("spark.jars", config['SPARKMOBILITY_JAR'])
             .config("spark.executor.memory", f"{config['MEMORY']}g")
             .config("spark.driver.memory", f"{config['MEMORY']}g")
             .config("spark.sql.files.ignoreCorruptFiles", "true")
             .config('spark.sql.session.timeZone', 'UTC')
             .getOrCreate())

    spark.sparkContext.setLogLevel(config['LOG_LEVEL'])

    log4jLogger = spark._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")
    return spark

def spark_session(func):
    """
    Decorator that creates a Spark session, passes it as the first argument
    to the decorated function, and stops the session when done.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        spark = create_spark_session()
        try:
            return func(spark, *args, **kwargs)
        finally:
            spark.stop()
    return wrapper

# Example usage of the decorator:
#
# @spark_session
# def processing_logic(spark, data):
#     # your function logic using the spark session
#     df = spark.read.json(data)
#     return df.count()
#
# The Spark session is automatically created before processing_logic is executed,
# and stopped afterwards.