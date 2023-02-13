from io import BytesIO
import tempfile
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import duckdb
import pyarrow.dataset as ds

from models.s3 import S3Client


FEATURES_ORDER = ["hour", "dow", "num_bikes_disabled", "num_docks_available", "num_docks_disabled"]
TARGET = "minutes_bt_check"
OHE_SLICE = [0, 1]
SS_SLICE = slice(2,8)
TEST_SIZE = 0.1

DATA_BASE_PATH = "silver/status/"
DATA_DATE_TEMPLATE = "year=%Y/month=%-m/day=%-d"

CURRENT_MODEL_KEY = "models/current_eta_model.txt"
JOBLIB_COMPRESSION = ('lzma', 3)
MODELS_BASE_PATH = "models/"
MODELS_DATE_TEMPLATE = "year=%Y/month=%m/%d"
MODELS_FILE_NAME = "_eta.joblib"
METRICS_FILE_NAME = "_eta_metrics.joblib"

BASE_TABLE_QUERY = """
WITH base_status AS (select
    station_id,
    hour,
    num_bikes_available,
    num_bikes_disabled,
    num_docks_available,
    num_docks_disabled,
    status,
    make_timestamp(year, month, day, hour, minute, 0.0) as ts,
from
    status
where
    station_id = {} and
    status = 'IN_SERVICE')"""
LEAD_MINUTES_QUERY = " UNION ".join(["""
SELECT
    station_id,
    hour,
    dayofweek(ts) as dow,
    num_bikes_available,
    num_bikes_disabled,
    num_docks_available,
    num_docks_disabled,
    minute(lead(ts, {}) over (
        order by ts asc
    ) - ts)  as minutes_bt_check,
    lead(num_bikes_available, {}) over (
        order by ts asc
    ) as bikes_available,
FROM
    base_status
""".format(i, i) for i in range(1, 16)])

class ETAModelTrainer:
    def __init__(self, features_order = FEATURES_ORDER, target = TARGET, ohe_slice = OHE_SLICE, ss_slice = SS_SLICE) -> None:
        self.features_order = features_order
        self.target = target
        self.ohe_slice = ohe_slice
        self.ss_slice = ss_slice
        self.s3_cli = S3Client()
        self.stations_pipelines = dict()
        self.stations_metrics = dict()

    def create_dataset(self, current_date, training_period = 100) -> pd.DataFrame:
        temp_dir = tempfile.TemporaryDirectory()
        dates = pd.date_range(end=current_date, periods=training_period)

        for day in dates:
            day_keys= self.s3_cli.client.Bucket(self.s3_cli.bucket).objects.filter(Prefix=DATA_BASE_PATH+day.strftime(DATA_DATE_TEMPLATE))
            for parquet_object in day_keys:
                parquet_temp_path = temp_dir.name + "/" + parquet_object.key
                os.makedirs(os.path.dirname(parquet_temp_path), exist_ok = True)
                self.s3_cli.client.Bucket(self.s3_cli.bucket).download_file(Key=parquet_object.key, Filename=parquet_temp_path)

        # Review this: https://duckdb.org/docs/guides/import/s3_import
        dataset = ds.dataset(temp_dir.name + "/silver/status", format="parquet", partitioning="hive")
        con = duckdb.connect()
        con = con.register("status", dataset)

        station_ids = con.execute("select distinct(station_id) from status").df()["station_id"].values
        dfs_to_concat = []

        for station_id in station_ids:
            df_query = BASE_TABLE_QUERY.format(station_id) + LEAD_MINUTES_QUERY
            dfs_to_concat.append(con.execute(df_query).df())
        dataset_df = pd.concat(dfs_to_concat)
        dataset_df = dataset_df[(dataset_df["num_bikes_available"] == 0) & (dataset_df["bikes_available"] > 0)]

        temp_dir.cleanup()
        return dataset_df

    def train_all_stations(self, data_set) -> None:
        for station_id in data_set["station_id"].unique():
            self.train_station(station_id, data_set[data_set["station_id"] == station_id])

    def train_station(self, station_id, data_set) -> None:
        station_pipeline = make_pipeline(ColumnTransformer([("ohe",  OneHotEncoder(sparse=False), [0, 1]), ("ss",  StandardScaler(), slice(2,5))]),
            MLPRegressor((128, 128, 128)))
        data_set = data_set.dropna()
        X_train, X_test, y_train, y_test = train_test_split(data_set[self.features_order].values, data_set[self.target].values, test_size=TEST_SIZE, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        station_pipeline.fit(X_train, y_train)
        self.stations_pipelines[station_id] = station_pipeline
        self.stations_metrics[station_id] = mean_absolute_error(y_test, station_pipeline.predict(X_test))

    def dump_stations_pipelines(self, date, current=False) -> None:
        model_key = MODELS_BASE_PATH + date.strftime(MODELS_DATE_TEMPLATE) + MODELS_FILE_NAME
        metrics_key = MODELS_BASE_PATH + date.strftime(MODELS_DATE_TEMPLATE) + METRICS_FILE_NAME
        with BytesIO() as mem_f:
            joblib.dump(self.stations_pipelines, mem_f, compress=JOBLIB_COMPRESSION)
            mem_f.seek(0)
            self.s3_cli.client.Bucket(self.s3_cli.bucket).upload_fileobj(Key=model_key, Fileobj=mem_f)
        with BytesIO() as mem_f:
            joblib.dump(self.stations_metrics, mem_f)
            mem_f.seek(0)
            self.s3_cli.client.Bucket(self.s3_cli.bucket).upload_fileobj(Key=metrics_key, Fileobj=mem_f)
        if current:
            self.s3_cli.client.Bucket(self.s3_cli.bucket).put_object(Key=CURRENT_MODEL_KEY, Body=model_key, ContentType= "text/plain")