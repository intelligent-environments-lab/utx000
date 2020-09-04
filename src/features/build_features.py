from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

class beacon_statistics():

    def __init__(self):
        pass

    def get_percent_completeness(self, df, start_time, end_time, sensor='CO2', beacon_no=-1):
        '''
        Gets the percent completeness for all beacons in the dataframe
        
        Parameters:
        - df: dataframe holding the beacon data with one column titled "Beacon"
        - start_time: datetime for the start of the analysis period
        - end_time: datetime for the end of the analysis period
        - sensor: string corresponding to the sensor variable to check for completeness
        - beacon_no: integer corresponding to the beacon of choice - default is -1 (all)
        
        Returns:
        - aggregate_completeness: dictionary with beacon number as key and percent completeness as 
          value
        - hourly_completeness: dictionary with beacon number as key and list of hourly percent
          compeleteness as values
        '''

        # vars to return
        aggregate_completeness = {}
        hourly_completeness = {}

        if beacon_no == -1:
            # all beacons - default
            beacon_list = df['Beacon'].unique()
        else:
            # list of just one beacon - must specify
            beacon_list = [beacon_no]

        # getting percent complete through list of desired beacons
        for beacon_no in beacon_list:
            data_by_id = df[df['Beacon'] == beacon_no]
            data_by_id_by_time = data_by_id[start_time:end_time]

            data_counts = data_by_id_by_time.resample(timedelta(hours=1)).count()
            # hourly completeness
            data_percentages = data_counts / 12
            hourly_completeness[beacon_no] = data_percentages

            # aggregate completeness
            overall_percentage = np.nansum(data_counts[sensor])/(len(data_by_id_by_time))
            aggregate_completeness[beacon_no] = overall_percentage

        return aggregate_completeness, hourly_completeness

    def get_measurement_time(self, df, start_time, end_time, sensor='CO2', threshold=0, below=True, beacon_no=-1, measurement_interval=5):
        '''
        Determine the number of measurements above or below certain threshold

        Parameters:
        - df: dataframe holding the beacon data with one column titled "Beacon"
        - start_time: datetime for the start of the analysis period
        - end_time: datetime for the end of the analysis period
        - sensor: string corresponding to the sensor variable to check for completeness
        - beacon_no: integer corresponding to the beacon of choice - default is -1 (all)
        - measurement_interval: integer specifying the typical time between measurements
        - threshold: integer or float specifying the value to compare against - default
          is zero
        - below: boolean specifying to look for values below (True) or above (False) the
          specified threshold

        Returns:
        - time: dictionary with keys as beacon numbers and the amount of time spent
          above/below a certain threshold the value
        '''
        # vars to return
        time = {}

        if beacon_no == -1:
            # all beacons - default
            beacon_list = df['Beacon'].unique()
        else:
            # list of just one beacon - must specify
            beacon_list = [beacon_no]

        # getting measurement times through list of desired beacons
        for beacon_no in beacon_list:
            data_by_id = df[df['Beacon'] == beacon_no]
            data_by_id_by_time = data_by_id[start_time:end_time]
            
            # counting the number of values above/below the threshold
            measurements = data_by_id_by_time[sensor].values
            if below:
                count = sum(map(lambda x : x<threshold, measurements))
            else:
                count = sum(map(lambda x : x>threshold, measurements))

            # adding result to dictionary
            time[beacon_no] = count*measurement_interval

        return time


