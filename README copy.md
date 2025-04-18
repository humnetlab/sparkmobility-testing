## Timegeo

## Pre-requisites
### Packaging the Scala Code

Whether modifying the Scala core of the project or running it, you must compile the Scala code and create a JAR file. Follow these steps:

1. **Download and Install SBT:**

    Install sbt on your machine. You can follow the instructions on the [official sbt documentation](https://www.scala-sbt.org/1.x/docs/Setup.html).

2. **Package the Scala Code:**

    Install,Compile and Package the Scala code using sbt:

    ```bash
    sbt update
    sbt compile
    sbt assembly
    ```
    This process creates a `.jar` file that should be submitted to the Spark cluster. Ideally, place it in the root directory of this project. You will need the PATH to this jar in the next steps.

## Setup and Installation of Spark

Sometimes, the default Scala version is 2.12. Since the project is compatible only with Scala 2.13 and above, you must set the Scala version before setting up the project. To do this, follow the instructions below:


1. **Download Spark:**
    - Go to the [Apache Spark download page](https://spark.apache.org/downloads.html).
    - Select the following options:
      - **Spark release:** 3.3.x or later
      - **Package type:** Pre-built for Apache Hadoop
      - **Scala version:** 2.13
    - Download the `.tgz` file.

2. **Extract the Spark Distribution:**
    Extract the downloaded file to a directory (e.g., `/opt/spark`):

    ```bash
    tar -xzf spark-3.5.4-bin-hadoop3-scala2.13.tgz -C /opt/spark
    ```

3. **Set Environment Variables:**
    Add the following environment variables to your shell configuration file (e.g., `.bashrc` or `.zshrc`):

    ```bash
    export SPARK_HOME=/opt/spark/spark-3.5.4-bin-hadoop3-scala2.13
    export PATH=$SPARK_HOME/bin:$PATH
    export PYSPARK_PYTHON=python3
    export PYSPARK_DRIVER_PYTHON=python3
    ```

4. **Reload the Shell Configuration:**

    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

5. **Install the Required Python Packages:**
    Install the required Python packages using `pip`:

    ```bash
    source env/bin/activate # activate the virtual environment
    pip install -r requirements.txt
    ```

6. **Verify the Scala Version:**
    Run the following in a Python shell or script:

    ```python
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    print(sc._jvm.scala.util.Properties.versionNumberString())
    ```
    This should output `2.13.x`.

If you are using conda follow this extract steps steps
1. **Use the Custom Spark Distribution in Conda:**
    - **Create a Conda Environment:**

      ```bash
      conda env create -f environment.yml
      ```

    - **Activate the Conda Environment:**

      ```bash
      conda activate environment_name
      ```

    - **Set SPARK_HOME in the Conda Environment:**

      ```bash
      conda env config vars set SPARK_HOME=/opt/spark/spark-3.5.4-bin-hadoop3-scala2.13
      ```

    - **Reactivate the Conda Environment:**

      ```bash
      conda deactivate
      conda activate environment_name
      ```

    - **Verify the Scala Version again:**

      ```python
      from pyspark import SparkContext
      sc = SparkContext.getOrCreate()
      print(sc._jvm.scala.util.Properties.versionNumberString())
      ```


### Using the Project

1. **Activate env**
    
    Activate the virtual environment:

    ```bash
    source env/bin/activate
    ```
    or
    ```bash
    conda activate environment_name
    ```

2. **Set environment variable**
    Set the following environment variable with the path to the jar file:

    ```bash
    export TIMEGEO_JAR=/path/to/jar/file
    ```

### Usage

This is a standard sbt project. You can use the following commands:

- Update dependencies: `sbt update`
- Compile the code: `sbt compile`
- Run the project: `sbt run`
- Start a Scala REPL: `sbt console`

### Changes on scala code

The scala code is located in the `src/main/scala` directory. You can modify the code in this directory.
Once edited you must follow the steps below to compile and package the code.

```
sbt update
sbt compile
sbt assembly
```
This process creates a .jar file that should be submitted to the Spark cluster. Ideally, place it in the root directory of this project.

Then you can replace the path to the jar file in the `spark-submit` command in the `run.sh` file.

For more information on sbt, visit the [official sbt documentation](https://www.scala-sbt.org/1.x/docs/).

