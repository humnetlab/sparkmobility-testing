from sparkmobility.utils.session import create_spark_session
class MobilityDataset:
    def __init__(self, 
                 dataset_name,
                 raw_data_path, 
                 column_mappings,
                processed_data_path=None,
                start_datetime="2020-01-01 00:00:00",
                end_datetime="2030-12-31 23:59:59",
                longitude=None,
                latitude=None,
                time_zone="America/Mexico_City",
                time_format="UNIX"
                ):
        self.dataset_name = dataset_name
        self.input_path = raw_data_path
        if processed_data_path is None:
            processed_data_path = f"{dataset_name}_processed"
        self.output_path = processed_data_path
        self.column_names = column_mappings
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.longitude = longitude
        self.latitude = latitude
        self.time_zone = time_zone
        self.time_format = time_format
        self.df = None

    def load_stays(self):
        try:
            df = create_spark_session().read.parquet(self.output_path + "/StayPointsWithHomeWork")
        except Exception as e:
            print(f"Error loading stays: {e}")
            return None
        return df
    

        