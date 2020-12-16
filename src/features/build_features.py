from datetime import datetime, timedelta

import os
import sys
import logging

import pandas as pd
import numpy as np

import geopy.distance

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

    def compare_temperature_readings(self, df):
        '''
        Compares temperature readings from the DGS and Sensirion sensors
        
        Parameters:
        - df: dataframe holding the beacon data with columns titled "T_NO2", "T_CO", and "Temperature [C]", "Beacon"
        
        Returns:
        - t_raw: dataframe holding the measured temperature values for all beacons
        - t_summary: dictionary with beacon numbers as keys and dataframe of statistical values for each t sensor
        '''
        
        t_raw = df[['T_NO2','T_CO','Temperature [C]','Beacon']]
        t_raw.columns = ['DGS1','DGS2','Sensirion','Beacon']
        def avg_dgs(x,y):
            if x < 0:
                return y
            elif y < 0:
                return x
            else:
                return (x+y)/2
            
        t_raw['DGS_AVG'] = t_raw.apply(lambda row: avg_dgs(row[0],row[1]),axis=1)
        t_raw['Difference'] = t_raw['Sensirion'] - t_raw['DGS_AVG']
        
        t_summary = {}
        for beacon in t_raw['Beacon'].unique():
            data_by_beacon = t_raw[t_raw['Beacon'] == beacon]
            means = [np.nanmean(data_by_beacon['DGS1']),np.nanmean(data_by_beacon['DGS2']),np.nanmean(data_by_beacon['Sensirion'])]
            mins = [np.nanmin(data_by_beacon['DGS1']),np.nanmin(data_by_beacon['DGS2']),np.nanmin(data_by_beacon['Sensirion'])]
            maxs = [np.nanmax(data_by_beacon['DGS1']),np.nanmax(data_by_beacon['DGS2']),np.nanmax(data_by_beacon['Sensirion'])]
            p25s = [np.nanpercentile(data_by_beacon['DGS1'],25),np.nanpercentile(data_by_beacon['DGS2'],25),np.nanpercentile(data_by_beacon['Sensirion'],25)]
            p75s = [np.nanpercentile(data_by_beacon['DGS1'],75),np.nanpercentile(data_by_beacon['DGS2'],75),np.nanpercentile(data_by_beacon['Sensirion'],75)]
        
            beacon_df = pd.DataFrame(data={'Min':mins,'Max':maxs,'Mean':means,'25th':p25s,'75th':p75s},
                  index=['DGS1', 'DGS2', 'Sensirion'])
            t_summary[beacon] = beacon_df
            
        return t_raw, t_summary

def get_restricted_beacon_datasets(radius=1000,restrict_by_ema=True, data_dir='../../'):
    '''
    Gets the most restricted/filtered dataset for the beacon considering we have fitbit,
    ema, and gps data for the night the participant slept.

    Inputs:
    - radius: the threshold to consider for the participants' GPS coordinates 

    Output:
    - filtered_beacon: dataframe holding filtered beacon data
    '''

    # Importing necessary processed data files
    # beacon data
    beacon = pd.read_csv(f'{data_dir}data/processed/bpeace2-beacon.csv',
                        index_col=0,parse_dates=True,infer_datetime_format=True)
    # fitbit sleep data
    sleep = pd.read_csv(f'{data_dir}data/processed/bpeace2-fitbit-sleep-daily.csv',
                    parse_dates=['date','startTime','endTime'],infer_datetime_format=True)
    end_dates = []
    for d in sleep['endTime']:
        end_dates.append(d.date())
    sleep['endDate'] = end_dates
    # EMA data
    ema = pd.read_csv(f'{data_dir}data/processed/bpeace2-morning-survey.csv',
                  index_col=0,parse_dates=True,infer_datetime_format=True)
    # gps data
    gps = pd.read_csv(f'{data_dir}data/processed/bpeace2-gps.csv',
                 index_col=0,parse_dates=[0,1],infer_datetime_format=True)
    # participant info data for beacon users
    info = pd.read_excel(f'{data_dir}data/raw/bpeace2/admin/id_crossover.xlsx',sheet_name='beacon',
                    parse_dates=[3,4,5,6],infer_datetime_format=True)


    nightly_beacon = pd.DataFrame() # df restricted by fitbit and gps
    for pt in sleep['beiwe'].unique():
        if pt in info['Beiwe'].values: # only want beacon particiapnts
            # getting data per participant
            gps_pt = gps[gps['Beiwe'] == pt]
            sleep_pt = sleep[sleep['beiwe'] == pt]
            beacon_pt = beacon[beacon['Beiwe'] == pt]
            info_pt = info[info['Beiwe'] == pt]
            lat_pt = info_pt['Lat'].values[0]
            long_pt = info_pt['Long'].values[0]
            print(f'Working for {pt} - Beacon', info_pt['Beacon'])
            # looping through sleep start and end times
            print(f'\tNumber of nights of sleep:', len(sleep_pt['startTime']))
            s = pd.to_datetime(np.nanmin(sleep_pt['endTime'])).date()
            e = pd.to_datetime(np.nanmax(sleep_pt['endTime'])).date()
            print(f'\tSpanning {s} to {e}')
            for start_time, end_time in zip(sleep_pt['startTime'],sleep_pt['endTime']):
                gps_pt_night = gps_pt[start_time:end_time]
                beacon_pt_night = beacon_pt[start_time:end_time]
                # checking distances between pt GPS and home GPS
                if len(gps_pt_night) > 0:
                    coords_1 = (lat_pt, long_pt)
                    coords_2 = (np.nanmean(gps_pt_night['Lat']), np.nanmean(gps_pt_night['Long']))
                    d = geopy.distance.distance(coords_1, coords_2).m
                    if d < radius:
                        # resampling so beacon and gps data are on the same time steps
                        gps_pt_night = gps_pt_night.resample('5T').mean()
                        beacon_pt_night = beacon_pt_night.resample('5T').mean()
                        nightly_temp = gps_pt_night.merge(right=beacon_pt_night,left_index=True,right_index=True,how='inner')
                        nightly_temp['start_time'] = start_time
                        nightly_temp['end_time'] = end_time
                        nightly_temp['Beiwe'] = pt

                        nightly_beacon = nightly_beacon.append(nightly_temp)
                        print(f'\tSUCCESS - added data for night {end_time.date()}')
                    else:
                        print(f'\tParticipant outside {radius} meters for night {end_time.date()}')
                else:
                    print(f'\tNo GPS data for night {end_time.date()}')
        else:
            print(f'{pt} did not receive a beacon')

    # Data filtered by fitbit nights only
    nightly_beacon.to_csv(f'{data_dir}data/processed/bpeace2-beacon-fb_and_gps_restricted.csv')

    if restrict_by_ema == True:
        # removing nights without emas the following morning 
        filtered_beacon = pd.DataFrame()
        for pt in nightly_beacon['Beiwe'].unique():
            # getting pt-specific dfs
            evening_iaq_pt = nightly_beacon[nightly_beacon['Beiwe'] == pt]
            ema_pt = ema[ema['ID'] == pt]
            survey_dates = ema_pt.index.date
            survey_only_iaq = evening_iaq_pt[evening_iaq_pt['end_time'].dt.date.isin(survey_dates)]
            
            filtered_beacon = filtered_beacon.append(survey_only_iaq)
            
        filtered_beacon.to_csv(f'{data_dir}data/processed/bpeace2-beacon-fb_ema_and_gps_restricted.csv')

    return nightly_beacon, filtered_beacon

def get_sleep_summaries(data_dir='../../'):
    '''
    Gets summary sleep data from EMAs and Fitbit for three datasets:
    - complete: all nights with both EMA and Fitbit recordings
    - fully filtered: nights when participants are home and have beacon data too

    Inputs:
    -

    Returns two dataframes pertaining to the sleep data for both datasets
    '''

    # Complete
    # --------
    print('\tGetting Complete Sleep Summary')
    # Self-report/EMA sleep
    ema_sleep = pd.read_csv(f'{data_dir}data/processed/bpeace2-morning-survey.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
    for column in ['TST','SOL','NAW','Restful']:
        ema_sleep = ema_sleep[ema_sleep[column].notna()]
    ema_sleep['date'] = pd.to_datetime(ema_sleep.index.date)
    # Fitbit-recorded sleep
    fb_sleep = pd.read_csv(f'{data_dir}data/processed/bpeace2-fitbit-sleep-daily.csv',index_col=0,parse_dates=True,infer_datetime_format=True)

    # Getting complete sleep dataframe
    complete_sleep = pd.DataFrame() # dataframe to append to
    pt_list = np.intersect1d(fb_sleep['beiwe'].unique(),ema_sleep['ID'].unique())
    for pt in pt_list:
        ema_sleep_beiwe = ema_sleep[ema_sleep['ID'] == pt]
        fb_sleep_beiwe = fb_sleep[fb_sleep['beiwe'] == pt]
        complete_sleep = complete_sleep.append(fb_sleep_beiwe.merge(ema_sleep_beiwe,left_on='date',right_on='date',how='inner'))
    # Saving
    complete_sleep.set_index('date',inplace=True)
    complete_sleep.to_csv(f'{data_dir}data/processed/bpeace2-fitbit_beiwe-complete_sleep_summary.csv')

    # Fully Filtered
    # --------------
    print('\tGetting Sleep Summary for Fully Filtered Beacon Dataset')
    # Fully filtered beacon dataset to cross-reference
    ff_beacon = pd.read_csv(f'{data_dir}data/processed/bpeace2-beacon-fb_ema_and_gps_restricted.csv',
                                 index_col=0, parse_dates=[0,-2,-1], infer_datetime_format=True)
    ff_beacon['date'] = ff_beacon['end_time'].dt.date

    # Getting fully filtered sleep dataframe
    ff_sleep = pd.DataFrame()
    for pt in ff_beacon['Beiwe'].unique():
        ff_sleep_pt = complete_sleep[complete_sleep['beiwe'] == pt]
        ff_pt = ff_beacon[ff_beacon['Beiwe'] == pt]
        ff_pt_summary = ff_pt.groupby('date').mean()

        ff_sleep = ff_sleep.append(ff_sleep_pt.merge(ff_pt_summary,left_index=True,right_index=True,how='inner'))
    # cleaning and saving
    ff_sleep.drop(['dateOfSleep','infoCode','logId','ID',
       'Lat', 'Long', 'Alt', 'Accuracy', 'TVOC', 'eCO2', 'Lux',
       'Visible', 'Infrared', 'NO2', 'T_NO2', 'RH_NO2', 'CO', 'T_CO', 'RH_CO',
       'Temperature [C]', 'Relative Humidity', 'CO2', 'PM_N_0p5', 'PM_N_1',
       'PM_N_2p5', 'PM_N_4', 'PM_N_10', 'PM_C_1', 'PM_C_2p5', 'PM_C_4',
       'PM_C_10',],axis=1,inplace=True)
    ff_sleep.to_csv(f'{data_dir}data/processed/bpeace2-fitbit_beiwe-fully_filtered_sleep_summary.csv')

    return complete_sleep, ff_sleep

def main():
    #get_restricted_beacon_datasets(data_dir='../../')
    get_sleep_summaries()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
