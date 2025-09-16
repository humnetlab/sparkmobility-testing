import functools

from pyspark.sql import SparkSession

from sparkmobility import config


def create_spark_session():
    print(config)
    spark = (
        SparkSession.builder.master(f"local[{config['CORES']}]")
        .appName("SparkMobility")
        .config("spark.jars", config["SPARKMOBILITY_JAR"])
        .config("spark.executor.memory", f"{config['MEMORY']}g")
        .config("spark.driver.memory", f"{config['MEMORY']}g")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", f"{config['TEMP_DIR']}")
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={config['TEMP_DIR']}")
        .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={config['TEMP_DIR']}")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.default.parallelism", "96")
        .config("spark.shuffle.io.maxRetries", "10")
        .config("spark.shuffle.io.retryWait", "5s")
        .config("spark.reducer.maxReqsInFlight", "1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(config["LOG_LEVEL"])

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
