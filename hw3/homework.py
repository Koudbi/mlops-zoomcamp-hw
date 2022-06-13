import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.flow_runners import SubprocessFlowRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task 
def get_path(date_=None):
    logger = get_run_logger("logger")
    date_format = "%Y-%m"
    path = "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata"
    if date_ is None :
        date_ = datetime.today()
    else:
        date_ = datetime.strptime(date, "%Y-%m-%d")
    
    train_date = (date_-relativedelta(months=2)).strftime(date_format)
    val_date = (date_-relativedelta(months=1)).strftime(date_format)

    train_path = f"{path}_{train_date}.parquet"
    logger.info(f"Train path is {train_path}")
    val_path = f"{path}_{val_date}.parquet"
    logger.info(f"Val path is {val_path}")

    return train_path, val_path

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger("logger")
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        #print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        #print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger("logger")
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    #print(f"The shape of X_train is {X_train.shape}")
    #print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    #print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger("logger")
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    #print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner)
#def main(train_path: str = './data/fhv_tripdata_2021-01.parquet', 
#           val_path: str = './data/fhv_tripdata_2021-02.parquet'):
def main(date = None):
    train_path, val_path = get_path(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f'models/model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

    with open(f'models/dv-{date}.pkl', 'wb') as f_out:
        pickle.dump(dv, f_out)


    run_model(df_val_processed, categorical, dv, lr)

#main()
#main(date="2021-08-15")

DeploymentSpec(
    flow=main,
    name="scheduled-9h15th-months",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner()
)

