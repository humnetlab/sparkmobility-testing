import json
import os

from sparkmobility.utils import spark_session


class StayDetection:
    def __init__(
        self,
        MobilityDataset,
    ):
        self.dataset = MobilityDataset

        os.makedirs(self.dataset.output_path, exist_ok=True)

        if os.path.exists(self.dataset.output_path + "/config.json"):
            with open(self.dataset.output_path + "/config.json", "r") as f:
                params = json.load(f)
        else:
            with open(self.dataset.output_path + "/config.json", "w") as f:
                params = {
                    "dataset_name": self.dataset.dataset_name,
                    "startTimestamp": self.dataset.start_datetime,
                    "endTimestamp": self.dataset.end_datetime,
                    "longitude": self.dataset.longitude,
                    "latitude": self.dataset.latitude,
                    "timeZone": self.dataset.time_zone,
                }
                json.dump(params, f, indent=4)

        self.param_file_path = os.path.abspath(
            self.dataset.output_path + "/config.json"
        )

    def _create_hashmap(self, spark):
        # hashmap = spark._jvm.java.util.HashMap()
        # for key, value in self.column_names.items():
        #     hashmap.put(key, value)
        hashmap = spark._jvm.PythonUtils.toScalaMap(self.dataset.column_names)
        return hashmap

    def _get_pipeline_instance(self, spark):
        jvm = spark._jvm
        return jvm.pipelines.Pipelines()

    @spark_session
    def get_stays(
        spark,
        self,
        delta_t=300,
        spatial_threshold=300,
        speed_threshold=6.0,
        temporal_threshold=300,
        hex_resolution=8,
        regional_temporal_threshold=3600,
        passing=True,
        home_to_work=8,
        work_to_home=19,
        work_distance_limit=500,
        work_freq_count_limit=3,
        find_home_and_work=True,
    ):
        with open(self.dataset.output_path + "/config.json", "r") as f:
            params = json.load(f)

        params.update(
            {
                "deltaT": delta_t,
                "spatialThreshold": spatial_threshold,
                "speedThreshold": speed_threshold,
                "temporalThreshold": temporal_threshold,
                "hexResolution": hex_resolution,
                "regionalTemporalThreshold": regional_temporal_threshold,
                "passing": passing,
                "homeToWork": home_to_work,
                "workToHome": work_to_home,
                "workDistanceLimit": work_distance_limit,
                "workFreqCountLimit": work_freq_count_limit,
                "findHomeAndWork": find_home_and_work,
            }
        )

        with open(self.dataset.output_path + "/config.json", "w") as f:
            json.dump(params, f, indent=4)

        pipeline = self._get_pipeline_instance(spark)
        columnNames = self._create_hashmap(spark)

        pipeline.getStays(
            self.dataset.input_path,
            self.dataset.output_path + "/StayPoints",
            self.dataset.time_format,
            "parquet",
            "",
            "true",
            columnNames,
            self.param_file_path,
        )
        if find_home_and_work:
            pipeline.getHomeWorkLocation(
                self.dataset.output_path + "/StayPoints",
                self.dataset.output_path + "/StayPointsWithHomeWork",
                self.param_file_path,
            )
            return "Stay detection completed with home and work locations labeled"
        else:
            return "Stay detection completed"

    @spark_session
    def get_home_work_od_matrix(spark, self, hex_resolution) -> None:
        pipeline = self._get_pipeline_instance(spark)
        with open(self.param_file_path, "r") as f:
            params = json.load(f)
        if hex_resolution > params["hexResolution"]:
            raise ValueError(
                f"Resolution must be smaller than {params['hexResolution']}"
            )
        pipeline.getODMatrix(
            self.dataset.output_path + "/StayPointsWithHomeWork",
            self.dataset.output_path
            + f"""/HomeWorkODMatrix/Resolution{hex_resolution}""",
            hex_resolution,
        )
        return "OD matrix based on home and work locations"

    @spark_session
    def get_od_matrix(spark, self, input_path, output_path, resolution):
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getFullODMatrix(input_path, output_path, resolution)
        return "od matrix based on default parameters"

    @spark_session
    def summarize(spark, self):
        with open(self.dataset.output_path + "/config.json", "r") as f:
            params = json.load(f)
        pipeline = self._get_pipeline_instance(spark)
        if params["findHomeAndWork"]:
            input_path = self.dataset.output_path + "/StayPointsWithHomeWork"
        else:
            input_path = self.dataset.output_path + "/StayPoints"

        pipeline.getStayDurationDistribution(
            input_path,
            self.dataset.output_path + "/Metrics/StayDurationDistribution",
        )

        # pipeline.getLocationDistribution(
        #     input_path,
        #     self.output_path + "/Metrics/LocationDistribution"
        # )

        pipeline.getDailyVisitedLocation(
            input_path,
            self.dataset.output_path + "/Metrics/DailyVisitedLocations",
        )
        pipeline.getDepartureTimeDistribution(
            input_path,
            self.dataset.output_path + "/Metrics/DepartureTimeDistribution",
        )

        return None
