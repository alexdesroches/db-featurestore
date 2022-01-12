# Databricks notebook source
# MAGIC %md
# MAGIC # Inference Functions

# COMMAND ----------

import os
os.environ['DATABRICKS_TOKEN'] =""

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone
from datetime import datetime
from pyspark.sql import *
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc


# COMMAND ----------

def read_raw_data():
  return spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")

@udf(returnType=IntegerType())
def is_weekend(dt):
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)  # 5 = Saturday, 6 = Sunday
  
@udf(returnType=StringType())  
def partition_id(dt):
    # datetime -> "YYYY-MM"
    return f"{dt.year:04d}-{dt.month:02d}"


def filter_df_by_ts(df, ts_column, start_date, end_date):
    if ts_column and start_date:
        df = df.filter(col(ts_column) >= start_date)
    if ts_column and end_date:
        df = df.filter(col(ts_column) < end_date)
    return df

def pickup_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the pickup_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df, ts_column, start_date, end_date
    )
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")
        )  # 1 hour window, sliding every 15 minutes
        .agg(
            mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        .select(
            col("pickup_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features
  
def dropoff_features_fn(df, ts_column, start_date, end_date):
    """
    Computes the dropoff_features feature group.
    To restrict features to a time range, pass in ts_column, start_date, and/or end_date as kwargs.
    """
    df = filter_df_by_ts(
        df,  ts_column, start_date, end_date
    )
    dropoffzip_features = (
        df.groupBy("dropoff_zip", window("tpep_dropoff_datetime", "30 minute"))
        .agg(count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            col("dropoff_zip").alias("zip"),
            unix_timestamp(col("window.end")).alias("ts").cast(IntegerType()),
            partition_id(to_timestamp(col("window.end"))).alias("yyyy_mm"),
            col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            is_weekend(col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features  


# COMMAND ----------

def get_sample_data(raw_data):
  
  pickup_features = pickup_features_fn(
      raw_data, ts_column="tpep_pickup_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
  )
  dropoff_features = dropoff_features_fn(
      raw_data, ts_column="tpep_dropoff_datetime", start_date=datetime(2016, 1, 1), end_date=datetime(2016, 1, 31)
  )
  
  return pickup_features, dropoff_features


# COMMAND ----------

def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).timestamp())

rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())

def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_pickup_datetime"], lit(15)),
        )
        .withColumn(
            "rounded_dropoff_datetime",
            rounded_unix_timestamp_udf(taxi_data_df["tpep_dropoff_datetime"], lit(30)),
        )
        .drop("tpep_pickup_datetime")
        .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df
  
def get_latest_model_version(model_name):
  latest_version = 1
  mlflow_client = MlflowClient()
  for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
    version_int = int(mv.version)
    if version_int > latest_version:
      latest_version = version_int
  return latest_version

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model_db_endpoint(dataset):
  #url = 'https://adb-5419940489217457.17.azuredatabricks.net/model/taxi_example_fare_packaged/3/invocations'
  url = "https://adb-5419940489217457.17.azuredatabricks.net/model/taxi_example_fare_packaged_aci/1/invocations"
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  print(data_json)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  
  print(response)
  return response, response.json()


def score_model_aci_endpoint(dataset):
  #url = 'https://adb-5419940489217457.17.azuredatabricks.net/model/taxi_example_fare_packaged/3/invocations'
  url = "http://26fd0e14-ad50-4497-a653-010704814f92.canadacentral.azurecontainer.io/score"
  # headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  headers = {}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  print(data_json)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  
  print(response)
  return response, response.json()

# COMMAND ----------

def create_scoring_dataset_withnometa():
  raw_data = read_raw_data()
  rounded_data = rounded_taxi_data(raw_data)
  cols = ['fare_amount', 'trip_distance', 'pickup_zip', 'dropoff_zip', 'rounded_pickup_datetime', 'rounded_dropoff_datetime']
  new_taxi_data_reordered = rounded_data.select(cols).alias("new")
  
  dropoff_feature_df = spark.table("feature_store_taxi_example.trip_dropoff_features").alias("dropoff")
  pickup_feature_df = spark.table("feature_store_taxi_example.trip_pickup_features").alias("pickup")
  
  taxi_data_and_features = (new_taxi_data_reordered.join(dropoff_feature_df, on=((dropoff_feature_df.ts == new_taxi_data_reordered.rounded_dropoff_datetime) 
                                                                              & (dropoff_feature_df.zip == new_taxi_data_reordered.dropoff_zip))
                                                                           , how='left')
                                                   .drop(*["zip", "ts","yyyy_mm","rounded_dropoff_datetime"])
                                                   .join(pickup_feature_df, on=((pickup_feature_df.ts == new_taxi_data_reordered.rounded_pickup_datetime) 
                                                                              & (pickup_feature_df.zip == new_taxi_data_reordered.pickup_zip))
                                                                           , how='left')
                                                   .drop(*["zip", "ts","yyyy_mm","rounded_pickup_datetime"])
                            .orderBy(new_taxi_data_reordered.fare_amount.desc()))
                           
  taxi_data_and_features.show(1)
  return taxi_data_and_features.drop('fare_amount').toPandas().head(1)
  

# COMMAND ----------

display(create_scoring_dataset_withnometa())

# COMMAND ----------

dataset = create_scoring_dataset_withnometa()

# COMMAND ----------

response, response_json = score_model_aci_endpoint(dataset)

# COMMAND ----------

response.text

# COMMAND ----------

response2, response_json2 = score_model_aci_endpoint(dataset)

# COMMAND ----------

response2.text

# COMMAND ----------


