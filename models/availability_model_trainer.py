from models.s3 import S3Client
from io import BytesIO
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



FEATURES_ORDER = ["hour", "dow", "num_bikes_available", "num_bikes_disabled", "num_docks_available", "num_docks_disabled", "minutes_bt_check"]
TARGET = "bikes_a"
CLASS_WEIGHT = {0: 1, 1: 500}
OHE_SLICE = [0, 1]
SS_SLICE = slice(2,7)
TEST_SIZE = 0.1

BUCKET = "frame"
CURRENT_MODEL_KEY = "models/current_availability_model.joblib"

class AvailabilityModelTrainer:
    def __init__(self, features_order = FEATURES_ORDER, target = TARGET, ohe_slice = OHE_SLICE, ss_slice = SS_SLICE) -> None:
        self.features_order = features_order
        self.target = target
        self.ohe_slice = ohe_slice
        self.ss_slice = ss_slice
        self.s3_cli = S3Client()
        self.stations_ids = []
        self.stations_pipelines = []
        self.stations_metrics = []

    def train_station(self, station_id, data_set) -> None:
        stations_pipeline = make_pipeline(
            ColumnTransformer([("ohe",  OneHotEncoder(sparse=False), self.ohe_slice), ("ss",  StandardScaler(), self.ss_slice)]),
            RandomForestClassifier(n_estimators=20, max_depth=40, class_weight=CLASS_WEIGHT))
        data_set = data_set.dropna()
        X_train, X_test, y_train, y_test = train_test_split(data_set[self.features_order].values, data_set[self.target].values, test_size=TEST_SIZE, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        self.stations_ids.append(station_id)
        stations_pipeline.fit(X_train, y_train)
        self.stations_pipelines.append(stations_pipeline)
        self.stations_metrics.append(confusion_matrix(y_test, stations_pipeline.predict(X_test), labels=[1, 0], normalize="true"))

    def dump_stations_pipelines(self, date, current=False):
        model_key = "models/"+ date.strftime('year=%Y/month=%m/%d') + "_availability.joblib"
        metrics_key = "models/"+ date.strftime('year=%Y/month=%m/%d') + "_availability_metrics.joblib"
        with BytesIO() as mem_f:
            joblib.dump([self.stations_ids, self.stations_pipelines], mem_f)
            mem_f.seek(0)
            self.s3_cli.client.Bucket(BUCKET).upload_fileobj(Key=model_key, Fileobj=mem_f)
            if current:
                model_to_copy = {
                    'Bucket': BUCKET,
                    'Key': model_key
                }
                self.s3_cli.client.Bucket(BUCKET).copy(model_to_copy, CURRENT_MODEL_KEY)
        with BytesIO() as mem_f:
            joblib.dump([self.stations_ids, self.stations_metrics], mem_f)
            mem_f.seek(0)
            self.s3_cli.client.Bucket(BUCKET).upload_fileobj(Key=metrics_key, Fileobj=mem_f)