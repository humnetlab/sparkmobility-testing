import configparser
import os


def load_config(config_path=None):
    config = configparser.ConfigParser()
    # Load default config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "spark_settings.ini")
    config.read(config_path)
    cfg = config["DEFAULT"]

    cores = int(os.getenv("SPARKMOBILITY_CORES", cfg.get("CORES", 50)))
    memory = int(os.getenv("SPARKMOBILITY_MEMORY", cfg.get("MEMORY", 64)))
    log_level = os.getenv("SPARKMOBILITY_LOG_LEVEL", cfg.get("LOG_LEVEL", "INFO"))
    timegeo_jar = os.getenv("SPARKMOBILITY_JAR", cfg.get("SPARKMOBILITY_JAR", "auto"))

    # Resolve JAR if set to auto
    if timegeo_jar.strip().lower() == "auto":
        timegeo_jar = find_jar()

    return {
        "CORES": cores,
        "MEMORY": memory,
        "LOG_LEVEL": log_level,
        "SPARKMOBILITY_JAR": timegeo_jar,
    }


def find_jar():
    jar_dir = os.path.join(os.path.dirname(__file__), "..", "lib")
    for f in os.listdir(jar_dir):
        if f.startswith("sparkmobility") and f.endswith(".jar"):
            return os.path.abspath(os.path.join(jar_dir, f))
    raise FileNotFoundError("Could not find sparkmobility jar in lib/")
