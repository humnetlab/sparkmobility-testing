from sparkmobility.utils import spark_session
import json
import os

class HomeAndWorkDetection:
    def __init__(self, param_file_path, timeZone="America/Mexico_City"):
        self.timeZone = timeZone
        self.param_file_path = param_file_path

    def _get_pipeline_instance(self, spark):
        jvm = spark._jvm
        return jvm.pipelines.Pipelines()
    
    @spark_session
    def get_home_and_work_locations(spark, self, input_path, output_path):
        """
        Detects home and work locations from stay data.
        
        :param spark: Spark session
        :param input_path: Path to the input stay data (output from stay detection)
        :param output_path: Path to save the detected home and work locations
        :return: None
        """
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getHomeWorkLocation(input_path, output_path, self.param_file_path)
        return None
    
    @spark_session
    def get_home_work_od_matrix(spark, self, input_path, output_path, resolution=8):
        """
        Generates an origin-destination matrix for home and work locations.
        
        :param spark: Spark session
        :param input_path: Path to the home and work locations data (output from home and work detection)
        :param output_path: Path to save the OD matrix
        :param resolution: Resolution for the hexagonal grid
        :return: None
        """
        pipeline = self._get_pipeline_instance(spark)
        pipeline.getODMatrix(input_path, output_path, resolution)
        return None
        

    
