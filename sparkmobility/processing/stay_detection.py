import json
import os

from sparkmobility.utils import spark_session


class StayDetection:
    def __init__(
        self,
        columns,
        input_path,
        output_path,
        startTimestamp="2020-01-01 00:00:00",
        endTimestamp="2025-12-31 23:59:59",
        longitude=None,
        latitude=None,
        timeZone="America/Mexico_City",
        timeFormat="UNIX",
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.column_names = columns
        self.timeFormat = timeFormat

        os.makedirs(output_path, exist_ok=True)

        if os.path.exists(output_path + "/config.json"):
            with open(output_path + "/config.json", "r") as f:
                params = json.load(f)
        else:
            with open(output_path + "/config.json", "w") as f:
                params = {
                    "startTimestamp": startTimestamp,
                    "endTimestamp": endTimestamp,
                    "longitude": longitude,
                    "latitude": latitude,
                    "timeZone": timeZone,
                }
                json.dump(params, f, indent=4)

        self.param_file_path = os.path.abspath(output_path + "/config.json")

    def _create_hashmap(self, spark):
        # hashmap = spark._jvm.java.util.HashMap()
        # for key, value in self.column_names.items():
        #     hashmap.put(key, value)
        hashmap = spark._jvm.PythonUtils.toScalaMap(self.column_names)
        return hashmap

    def _get_pipeline_instance(self, spark):
        jvm = spark._jvm
        return jvm.pipelines.Pipelines()

    @spark_session
    def get_stays(
        spark,
        self,
        deltaT=300,
        spatialThreshold=300,
        speedThreshold=6.0,
        temporalThreshold=300,
        hexResolution=8,
        regionalTemporalThreshold=3600,
        passing=True,
        homeToWork=8,
        workToHome=19,
        workDistanceLimit=500,
        workFreqCountLimit=3,
        findHomeAndWork=True,
    ):
        with open(self.output_path + "/config.json", "r") as f:
            params = json.load(f)

        params.update(
            {
                "deltaT": deltaT,
                "spatialThreshold": spatialThreshold,
                "speedThreshold": speedThreshold,
                "temporalThreshold": temporalThreshold,
                "hexResolution": hexResolution,
                "regionalTemporalThreshold": regionalTemporalThreshold,
                "passing": passing,
                "homeToWork": homeToWork,
                "workToHome": workToHome,
                "workDistanceLimit": workDistanceLimit,
                "workFreqCountLimit": workFreqCountLimit,
                "findHomeAndWork": findHomeAndWork,
            }
        )

        with open(self.output_path + "/config.json", "w") as f:
            json.dump(params, f, indent=4)

        pipeline = self._get_pipeline_instance(spark)
        columnNames = self._create_hashmap(spark)

        pipeline.getStays(
            self.input_path,
            self.output_path + "/StayPoints",
            self.timeFormat,
            "parquet",
            "",
            "true",
            columnNames,
            self.param_file_path,
        )
        if findHomeAndWork:
            pipeline.getHomeWorkLocation(
                self.output_path + "/StayPoints",
                self.output_path + "/StayPointsWithHomeWork",
                self.param_file_path,
            )
            return "Stay detection completed with home and work locations labeled"
        else:
            return "Stay detection completed"

    @spark_session
    def get_home_work_od_matrix(spark, self, resolution) -> None:
        pipeline = self._get_pipeline_instance(spark)
        with open(self.param_file_path, "r") as f:
            params = json.load(f)
        if resolution > params["hexResolution"]:
            raise ValueError(
                f"Resolution must be smaller than {params['hexResolution']}"
            )
        pipeline.getODMatrix(
            self.output_path + "/StayPointsWithHomeWork",
            self.output_path + f"""/HomeWorkODMatrix/Resolution{resolution}""",
            resolution,
        )
        return "OD matrix based on home and work locations"

    @spark_session
    def get_od_matrix(spark, self, input_path, output_path, resolution):
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getFullODMatrix(input_path, output_path, resolution)
        return "od matrix based on default parameters"

    @spark_session
    def summarize(spark, self):
        with open(self.output_path + "/config.json", "r") as f:
            params = json.load(f)
        pipeline = self._get_pipeline_instance(spark)
        if params["findHomeAndWork"]:
            input_path = self.output_path + "/StayPointsWithHomeWork"
        else:
            input_path = self.output_path + "/StayPoints"

        pipeline.getStayDurationDistribution(
            input_path,
            self.output_path + "/Metrics/StayDurationDistribution",
        )

        # pipeline.getLocationDistribution(
        #     input_path,
        #     self.output_path + "/Metrics/LocationDistribution"
        # )

        pipeline.getDailyVisitedLocation(
            input_path,
            self.output_path + "/Metrics/DailyVisitedLocations",
        )
        pipeline.getDepartureTimeDistribution(
            input_path,
            self.output_path + "/Metrics/DepartureTimeDistribution",
        )

        return None
