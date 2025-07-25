import os
import urllib.request

JAR_URL = "https://storage.googleapis.com/sparkmobility/sparkmobility010.jar"
JAR_NAME = "sparkmobility010.jar"
JAR_PATH = os.path.join(os.path.dirname(__file__), "lib", JAR_NAME)


def ensure_jar():
    if not os.path.exists(JAR_PATH):
        print(f"JAR file not found at {JAR_PATH}. Downloading from GCS...")
        os.makedirs(os.path.dirname(JAR_PATH), exist_ok=True)
        urllib.request.urlretrieve(JAR_URL, JAR_PATH)
        print("Download complete.")


ensure_jar()

from .settings import config, configure_env, download_and_extract

download_and_extract()
configure_env()
