# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:34:12 2021

@author: zphelan
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
    df = df[df['Pump Pressure'] > 0]
    df = df[df['Pump Pressure'] < 10_000]
    df = df[df['Differential Pressure'] < 8000]
    # df = df[df['Flow In'] < 1_000]
    df = df[df['Top Drive Torque (ft-lbs)'] > 0]
    df = df[df['Weight on Bit'] > 0]
    df = df[df['Weight on Bit'] < 100]
    df = df[df['Bit RPM'] < 500]
    return df


def remove_stopped(df, var='Total Depth'):
    df.drop_duplicates(inplace=True)
    # Base cases
    df = df[df < 12000]
    df = df[df >= 0]
    temp = df.reset_index()

    vals = temp[var].values

    # Remove depth values that are decreasing, those are impossible
    to_drop = []
    prev = 0
    for i in range(1, vals.size):
        if vals[i] <= vals[prev]:
            to_drop.append(temp['index'][i])

        else:
            prev = i

    df.drop(to_drop, axis=0, inplace=True)
    return df
    

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
