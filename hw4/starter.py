#!/usr/bin/env python
# coding: utf-8



#get_ipython().system('pip freeze | grep scikit-learn')
# In[ ]:


import pickle
import argparse
import pandas as pd
import sys
import numpy as np

# In[ ]:

# In[ ]:


def read_data(filename,categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[ ]:

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year",
        default="2021",type=int
    )
    parser.add_argument("--month",
        default="01",type=int
    )
    args = parser.parse_args()

    year = args.year
    month = args.month 

    categorical = ['PUlocationID', 'DOlocationID']  
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet',categorical) 
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    
    print(np.mean(y_pred))

    #df_result = pd.DataFrame()
    #df_result['prediction'] = y_pred
    #df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')

# In[ ]:




