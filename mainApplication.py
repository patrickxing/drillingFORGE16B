# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:27:35 2023

@author: xingp
"""

from preprocessing import remove_stopped, base_filter, calculateROP
import pandas
import graphing
import seaborn


def main():
    # load the data set
    """ when combine the data and time, we should use 
        CONCATENATE(TEXT(A2,"mm-dd-yyyy")," ",TEXT(B2,"hh:mm:ss"))
    """
    data = pandas.read_csv('../drillingData16bRaw_05012023.csv', parse_dates=['Time'], \
                         date_parser=lambda x: pandas.to_datetime(x, format='%m-%d-%Y %H:%M:%S'), \
                             low_memory=False)
    
    # apply the base filter
    dataBaseFilter = base_filter(data)
    
    # get a copy of data afer base filtered
    dataBaseFilterCopy = dataBaseFilter.copy()
    
    # keep only the increase hole depth
    # Note after remove_stopped, the index is kept
    depthIncrease = remove_stopped(dataBaseFilterCopy['Hole Depth (feet)'], 'Hole Depth (feet)')
    
    # drop the original depth that has non-increasing data
    dataBaseFilter.drop(["Hole Depth (feet)"], axis=1, inplace=True)
    
    
    # combine the increase only depth and the other columns in datBaseFilter
    dataBaseFilterIncreaseDepth = dataBaseFilter.join(depthIncrease)
    # remove the empties 
    dataBaseFilterIncreaseDepth.dropna(inplace = True)
    
    # calculate the ROP
    dataBaseFilterIncreaseDepthROP = calculateROP(dataBaseFilterIncreaseDepth)
    
    # clean up the data frame for machine learning analysis
    # columns we care about machine learning models
    MLVars = ['Weight on Bit (klbs)', 'Top Drive Torque (kft_lb)',
           'Differential Pressure (psi)', 'Standpipe Pressure (psi)', 'Top Drive Rotary (RPM)', 'Bit Torque (kft_lb)', 'Bit RPM (RPM)', "Calculated ROP (ft/hr) SMA60"]
    dataBaseFilterIncreaseDepthML = dataBaseFilterIncreaseDepthROP.loc[:, MLVars]
    
    # Graph correlation between variables
    seaborn.set_context("paper", font_scale=1.0)
    graphing.graphCorrelation(dataBaseFilterIncreaseDepthML,"Base Filter Data Correlation")
    
    print("this is break point")

    #data = pandas.read_csv('../drillingData16bRaw_05012023.csv', stringsAsFactors=false, low_memory=False)
    
    

main()