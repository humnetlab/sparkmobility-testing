from sparkmobility.utils import spark_session
import json
import os

class Stays:
    def __init__(self, deltaT=300, spatialThreshold=300, speedThreshold=6.0, temporalThreshold=300,
                 hexResolution=8, regionalTemporalThreshold=3600, passing=True,
                 startTimestamp="2022-11-01 10:50:30", endTimestamp="2023-02-02 12:20:45",
                 longitude=None, latitude=None, homeToWork=8, workToHome=19,
                 workDistanceLimit=500, workFreqCountLimit=3, timeZone="America/Mexico_City", 
                 timeFormat="UNIX", columns={}):
        if longitude is None:
            longitude = [-99.3, -98.7]
        if latitude is None:
            latitude = [19.2, 19.7]
        self.deltaT = deltaT
        self.spatialThreshold = spatialThreshold
        self.speedThreshold = speedThreshold
        self.temporalThreshold = temporalThreshold
        self.hexResolution = hexResolution
        self.regionalTemporalThreshold = regionalTemporalThreshold
        self.passing = passing
        self.startTimestamp = startTimestamp
        self.endTimestamp = endTimestamp
        self.longitude = longitude
        self.latitude = latitude
        self.homeToWork = homeToWork
        self.workToHome = workToHome
        self.workDistanceLimit = workDistanceLimit
        self.workFreqCountLimit = workFreqCountLimit
        self.timeZone = timeZone
        self.timeFormat = timeFormat
        self.column_names = columns
        self.params_file = self._create_parameters_file("./parameters.json")

    def _create_parameters_file(self, filepath):
        with open(filepath, "w") as f:
            params = {
                "deltaT": self.deltaT,
                "spatialThreshold": self.spatialThreshold,
                "speedThreshold": self.speedThreshold,
                "temporalThreshold": self.temporalThreshold,
                "hexResolution": self.hexResolution,
                "regionalTemporalThreshold": self.regionalTemporalThreshold,
                "passing": self.passing,
                "startTimestamp": self.startTimestamp,
                "endTimestamp": self.endTimestamp,
                "longitude": self.longitude,
                "latitude": self.latitude,
                "homeToWork": self.homeToWork,
                "workToHome": self.workToHome,
                "workDistanceLimit": self.workDistanceLimit,
                "workFreqCountLimit": self.workFreqCountLimit,
                "timeZone": self.timeZone
            }
            json.dump(params, f, indent=4)
        return os.path.abspath(filepath)
    
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
    def get_stays(spark, self, input_path, output_path):
        pipeline = self._get_pipeline_instance(spark)
        columnNames = self._create_hashmap(spark)
        print(type(columnNames))
        pipeline.getStays(input_path, output_path, self.timeFormat, columnNames, self.params_file)
        return "stays data based on default parameters"

    @spark_session
    def get_home_work_locations(spark, self, input_path, output_path):
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getHomeWorkLocation(input_path, output_path, self.params_file)
        return "home and work location based on default parameters"
    
    @spark_session
    def get_od_matrix(spark, self, input_path, output_path, resolution=8):
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getODMatrix(input_path, output_path, resolution)
        return "od matrix based on default parameters"


