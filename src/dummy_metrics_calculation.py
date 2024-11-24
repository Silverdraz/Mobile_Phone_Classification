import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

import data_preprocessing
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""
DATA_PATH = r"..\data\MobilePriceClassification" #Path to raw data
MODEL_PATH = r"..\models" #Path to models 

reference_data = pd.read_csv(os.path.join(DATA_PATH,"train.csv"))
with open(os.path.join(MODEL_PATH,"model.pkl"), 'rb') as f_in:
	model = joblib.load(f_in)

current_data = pd.read_csv(os.path.join(DATA_PATH,"test.csv"))

num_features = ['battery_power', 'clock_speed', 'fc', 'int_memory','m_dep','mobile_wt','n_cores','pc','px_height',
				'px_width','ram','sc_h','sc_w','talk_time']
cat_features = ['old','blue', 'dual_sim','four_g','three_g','touch_screen','wifi']

column_mapping = ColumnMapping(
    prediction='price_range',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='price_range'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)
	
@task(cache_policy=None)
def calculate_metrics_postgresql(curr,reference_data,current_data):
	#For applying functions on both dataframes
	combined_df = [reference_data,current_data]	
	#Remove unique identifier column
	reference_data, current_data = data_preprocessing.remove_columns(combined_df)
	combine_x = [reference_data,current_data]
	reference_data, current_data = data_preprocessing.feature_engineering_old(combine_x)
	
	current_data['price_range'] = model.predict(current_data)
	
	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)
	result = report.as_dict()
	
	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

	curr.execute(
		"insert into dummy_metrics(prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s)",
		( prediction_drift, num_drifted_columns, share_missing_values)
	)
    
@flow
def batch_monitoring_backfill(reference_data,current_data):
	prep_db()
	#last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr,reference_data,current_data )
			logging.info("data sent")

if __name__ == '__main__':
	#reference_data = pd.read_csv(os.path.join(DATA_PATH,"test.csv"))
	reference_data = pd.read_csv(os.path.join(DATA_PATH,"train.csv"))
	current_data = pd.read_csv(os.path.join(DATA_PATH,"test.csv"))
	
	batch_monitoring_backfill(reference_data,current_data)