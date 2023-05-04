# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:34:12 2021

@author: zphelan
Modified by Pengju Xing
"""
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def preprocess(df, filtering='z', zthres=3):
    df_filtered = df.dropna()  # Remove rows with missing data.
    df_filtered = base_filter(df_filtered)
    # Filter data with specified type.
    #df_filtered = filter_df(df_filtered, filtering, zthres)

    return df_filtered


# Remove values explicitly that should not be in the data frame.
def base_filter(df):
    df = df[df['Top Drive Torque (kft_lb)'] > 0]
    df = df[df['Top Drive Torque (kft_lb)'] < 100]
    df = df[df['Standpipe Pressure (psi)'] > 0]
    df = df[df['Standpipe Pressure (psi)'] < 10_000]
    df = df[df['Bit RPM (RPM)'] > 0]
    df = df[df['Bit RPM (RPM)'] < 500]
    
    df = df[df['Top Drive Rotary (RPM)'] > 0]
    df = df[df['Top Drive Rotary (RPM)'] < 500]
    

    df = df[df['Weight on Bit (klbs)'] > 0]
    df = df[df['Weight on Bit (klbs)'] < 200]
    
    df = df[df['Differential Pressure (psi)'] < 8000]
    
    df = df[df['Bit Torque (kft_lb)'] > 0]
    df = df[df['Bit Torque (kft_lb)'] < 100]
    
    return df


def remove_stopped(df, var):
    # drop_duplicates return DataFrame with duplicate rows removed.
    # inplace is True means modify the current dataframe
    df.drop_duplicates(inplace=True)
    # Base cases
    df = df[df < 12000]
    df = df[df >= 0]
    
    # Here reset_index does not modify the existing df, but return a new dataframe (temp) 
    temp = df.reset_index()

    vals = temp[var].values

    # Remove depth values that are decreasing or the same, those are impossible
    to_drop = []
    prev = 0
    for i in range(1, vals.size):
        if vals[i] <= vals[prev]:
            to_drop.append(temp['index'][i])

        else:
            prev = i
            
    # axis = 0 means index
    df.drop(to_drop, axis=0, inplace=True)
    return df

def calculateROP(dataBaseFilterIncreaseDepth):
    time = dataBaseFilterIncreaseDepth['Time']
    depth = dataBaseFilterIncreaseDepth['Hole Depth (feet)'] 
    shiftedTime = time.shift(1)
    shiftedDepth = depth.shift(1)
    dataBaseFilterIncreaseDepth['Time Diff (sec)'] = time.values - shiftedTime.values
    dataBaseFilterIncreaseDepth['Depth Diff (ft)'] = depth.values - shiftedDepth.values
    dataBaseFilterIncreaseDepth.dropna(inplace=True)
    # convert the time difference to be seconds
    dataBaseFilterIncreaseDepth['Time Diff (sec)'] = dataBaseFilterIncreaseDepth['Time Diff (sec)'].dt.total_seconds().astype(int)
    dataBaseFilterIncreaseDepth['Calculated ROP (ft/hr)'] = dataBaseFilterIncreaseDepth['Depth Diff (ft)'] / dataBaseFilterIncreaseDepth['Time Diff (sec)'] * 3600
    # moving average of the ROP
    dataBaseFilterIncreaseDepth['Calculated ROP (ft/hr) SMA60'] = dataBaseFilterIncreaseDepth['Calculated ROP (ft/hr)'].rolling(60).mean()
    dataBaseFilterIncreaseDepth.dropna(inplace=True)
    return dataBaseFilterIncreaseDepth

def filter_df(df, filtering='iqr', zthres=3):
    filtering = filtering.lower()  # standardize to lowercase
    # Filter from data frame using IQR boundaries
    # Note that IQR does not perform well with multivariate functions
    # Because our data set has 5 variables currently, it would be better to use a different method.
    if filtering == 'iqr':
        # IQR = stats.iqr(df.values)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_filtered = df[ ~((df < (Q1 - 1.5 * IQR)) |
                           (df > (Q3 + 1.5 * IQR)))]

        # Filter from data frame using z-score
        # Note that our data set is not normal or univariate, so z-score would not make sense here.
    elif filtering == 'zscore' or filtering == 'z':
        z = stats.zscore(df)
        z = np.abs(z)
        df_filtered = df[(z < zthres).all(axis=1)]

    else:
        print('Error: filtering type not found. Options: IQR, zscore.')
        raise ValueError

    return df_filtered


def mv_filter_df(train_X, train_y, filtering='isolation', random_state=None):
    # y_train = y_train.to_frame(name='ROP Calculated')
    print('train_X shape before = ' + str(train_X.shape) + ', y shape before = ' + str(train_y.shape))
    if filtering == 'isolation':
        iso = IsolationForest(random_state=random_state)
        y_hat = iso.fit_predict(train_X[train_X.columns])

    if filtering == 'lof':
        lof = LocalOutlierFactor()
        y_hat = lof.fit_predict(train_X)

    train_X.loc[:, 'outliers'] = y_hat
    train_y.loc[:, 'outliers'] = y_hat

    train_X = train_X[train_X['outliers'] > 0]
    train_y = train_y[train_y['outliers'] > 0]

    train_X.drop(['outliers'], axis=1, inplace=True)
    train_y.drop(['outliers'], axis=1, inplace=True)
    return train_X, train_y
