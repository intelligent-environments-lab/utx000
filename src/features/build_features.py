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

    def get_percent_completeness(self, df, start_time, end_time, sensor='co2', beacon_no=-1):
        '''
        Gets the percent completeness for all beacons in the dataframe
        
        Parameters:
        - df: dataframe holding the beacon data with one column titled "beacon"
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
            beacon_list = df['beacon'].unique()
        else:
            # list of just one beacon - must specify
            beacon_list = [beacon_no]

        # getting percent complete through list of desired beacons
        for beacon_no in beacon_list:
            data_by_id = df[df['beacon'] == beacon_no]
            data_by_id_by_time = data_by_id[start_time:end_time]

            data_counts = data_by_id_by_time.resample(timedelta(hours=1)).count()
            # hourly completeness
            data_percentages = data_counts / 12
            hourly_completeness[beacon_no] = data_percentages

            # aggregate completeness
            overall_percentage = np.nansum(data_counts[sensor])/(len(data_by_id_by_time))
            aggregate_completeness[beacon_no] = overall_percentage

        return aggregate_completeness, hourly_completeness

    def get_measurement_time(self, df, start_time, end_time, sensor='co2', threshold=0, below=True, beacon_no=-1, measurement_interval=5):
        '''
        Determine the number of measurements above or below certain threshold

        Parameters:
        - df: dataframe holding the beacon data with one column titled "beacon"
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
            beacon_list = df['beacon'].unique()
        else:
            # list of just one beacon - must specify
            beacon_list = [beacon_no]

        # getting measurement times through list of desired beacons
        for beacon_no in beacon_list:
            data_by_id = df[df['beacon'] == beacon_no]
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

def get_restricted_beacon_datasets(radius=1000,restrict_by_ema=True,data_dir='../../',study_suffix="ux_s20"):
    '''
    Gets restricted/filtered datasets for the beacon considering we have fitbit/GPS alone
    or fitbit, ema, and gps data for the night the participant slept.

    Inputs:
    - radius: the threshold to consider for the participants' GPS coordinates 
    - restrict_by_ema: boolean to control whether or not we create a second dataset filtered by ema
    - data_dir: string corresponding to the location of the "data" dir
    - study_suffix: string used to find the file and save the new files

    Output:
    - partially_filtered_beacon: dataframe with beacon data filtered by fitbit and gps measurements
    - fully_filtered_beacon: dataframe with beacon data filtered by fitbit, ema, and gps measurements
    '''

    # Importing necessary processed data files
    # beacon data
    beacon = pd.read_csv(f'{data_dir}data/processed/beacon-{study_suffix}.csv',
                        index_col=0,parse_dates=True,infer_datetime_format=True)
    # fitbit sleep data
    sleep = pd.read_csv(f'{data_dir}data/processed/fitbit-sleep_daily-{study_suffix}.csv',
                    parse_dates=['date','start_time','end_time'],infer_datetime_format=True)
    end_dates = []
    for d in sleep['end_time']:
        end_dates.append(d.date())
    sleep['end_date'] = end_dates
    # EMA data
    ema = pd.read_csv(f'{data_dir}data/processed/beiwe-morning_ema-{study_suffix}.csv',
                  index_col=0,parse_dates=True,infer_datetime_format=True)
    # gps data
    gps = pd.read_csv(f'{data_dir}data/processed/beiwe-gps-{study_suffix}.csv',
                 index_col=0,parse_dates=[0,1],infer_datetime_format=True)
    # participant info data for beacon users
    info = pd.read_excel(f'{data_dir}data/raw/utx000/admin/id_crossover.xlsx',sheet_name='beacon',
                    parse_dates=[3,4,5,6],infer_datetime_format=True)


    partially_filtered_beacon = pd.DataFrame() # df restricted by fitbit and gps
    for pt in sleep['beiwe'].unique():
        if pt in info['beiwe'].values: # only want beacon particiapnts
            # getting data per participant
            gps_pt = gps[gps['beiwe'] == pt]
            sleep_pt = sleep[sleep['beiwe'] == pt]
            beacon_pt = beacon[beacon['beiwe'] == pt]
            info_pt = info[info['beiwe'] == pt]
            lat_pt = info_pt['lat'].values[0]
            long_pt = info_pt['long'].values[0]
            print(f'Working for {pt} - Beacon', info_pt['beacon'])
            # looping through sleep start and end times
            print(f'\tNumber of nights of sleep:', len(sleep_pt['start_time']))
            s = pd.to_datetime(np.nanmin(sleep_pt['end_time'])).date()
            e = pd.to_datetime(np.nanmax(sleep_pt['end_time'])).date()
            print(f'\tSpanning {s} to {e}')
            for start_time, end_time in zip(sleep_pt['start_time'],sleep_pt['end_time']):
                gps_pt_night = gps_pt[start_time:end_time]
                beacon_pt_night = beacon_pt[start_time:end_time]
                # checking distances between pt GPS and home GPS
                if len(gps_pt_night) > 0:
                    coords_1 = (lat_pt, long_pt)
                    coords_2 = (np.nanmean(gps_pt_night['lat']), np.nanmean(gps_pt_night['long']))
                    d = geopy.distance.distance(coords_1, coords_2).m
                    if d < radius:
                        # resampling so beacon and gps data are on the same time steps
                        gps_pt_night = gps_pt_night.resample('5T').mean()
                        beacon_pt_night = beacon_pt_night.resample('5T').mean()
                        nightly_temp = gps_pt_night.merge(right=beacon_pt_night,left_index=True,right_index=True,how='inner')
                        nightly_temp['start_time'] = start_time
                        nightly_temp['end_time'] = end_time
                        nightly_temp['beiwe'] = pt

                        partially_filtered_beacon = partially_filtered_beacon.append(nightly_temp)
                        print(f'\tSUCCESS - added data for night {end_time.date()}')
                    else:
                        print(f'\tParticipant outside {radius} meters for night {end_time.date()}')
                else:
                    print(f'\tNo GPS data for night {end_time.date()}')
        else:
            print(f'{pt} did not receive a beacon')

    # Data filtered by fitbit nights only
    partially_filtered_beacon.to_csv(f'{data_dir}data/processed/beacon-fb_and_gps_filtered-{study_suffix}.csv')

    if restrict_by_ema == True:
        # removing nights without emas the following morning 
        fully_filtered_beacon = pd.DataFrame()
        for pt in partially_filtered_beacon['beiwe'].unique():
            # getting pt-specific dfs
            evening_iaq_pt = partially_filtered_beacon[partially_filtered_beacon['beiwe'] == pt]
            ema_pt = ema[ema['beiwe'] == pt]
            survey_dates = ema_pt.index.date
            survey_only_iaq = evening_iaq_pt[evening_iaq_pt['end_time'].dt.date.isin(survey_dates)]
            
            fully_filtered_beacon = fully_filtered_beacon.append(survey_only_iaq)
            
        fully_filtered_beacon.to_csv(f'{data_dir}data/processed/beacon-fb_ema_and_gps_filtered-{study_suffix}.csv')

    return partially_filtered_beacon, fully_filtered_beacon

def get_sleep_summaries(data_dir='../../',study_suffix="ux_s20"):
    '''
    Gets summary sleep data from EMAs and Fitbit

    Inputs:
    - data_dir: string corresponding to the location of the "data" dir
    - study_suffix: string used to find the file and save the new files

    Returns two dataframes pertaining to the sleep data for both datasets
    - complete: all nights with both EMA and Fitbit recordings
    - fully filtered: nights when participants are home and have beacon data too
    '''

    # Complete
    # --------
    print('\tGetting Complete Sleep Summary')
    # Self-report/EMA sleep
    ema_sleep = pd.read_csv(f'{data_dir}data/processed/beiwe-morning_ema-{study_suffix}.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
    for column in ['tst','sol','naw','restful']:
        ema_sleep = ema_sleep[ema_sleep[column].notna()]
    ema_sleep['date'] = pd.to_datetime(ema_sleep.index.date)
    # Fitbit-recorded sleep
    fb_sleep = pd.read_csv(f'{data_dir}data/processed/fitbit-sleep_daily-{study_suffix}.csv',index_col="date",parse_dates=True,infer_datetime_format=True)

    # Getting complete sleep dataframe
    complete_sleep = pd.DataFrame() # dataframe to append to
    pt_list = np.intersect1d(fb_sleep['beiwe'].unique(),ema_sleep['beiwe'].unique())
    for pt in pt_list:
        ema_sleep_beiwe = ema_sleep[ema_sleep['beiwe'] == pt]
        fb_sleep_beiwe = fb_sleep[fb_sleep['beiwe'] == pt]
        complete_sleep = complete_sleep.append(fb_sleep_beiwe.merge(ema_sleep_beiwe,left_on='date',right_on='date',how='inner'))
    # Saving
    complete_sleep.set_index('date',inplace=True)
    complete_sleep["beiwe"] = complete_sleep["beiwe_x"]
    complete_sleep.drop(["beiwe_x","beiwe_y"],axis=1,inplace=True)
    complete_sleep.to_csv(f'{data_dir}data/processed/fitbit_beiwe-sleep_summary-{study_suffix}.csv')

    # Fully Filtered
    # --------------
    print('\tGetting Sleep Summary for Fully Filtered Beacon Dataset')
    # Fully filtered beacon dataset to cross-reference
    ff_beacon = pd.read_csv(f'{data_dir}data/processed/beacon-fb_ema_and_gps_filtered-{study_suffix}.csv',
                                 index_col=0, parse_dates=[0,-2,-1], infer_datetime_format=True)
    ff_beacon['date'] = ff_beacon['end_time'].dt.date

    # Getting fully filtered sleep dataframe
    ff_sleep = pd.DataFrame()
    for pt in ff_beacon['beiwe'].unique():
        ff_sleep_pt = complete_sleep[complete_sleep['beiwe'] == pt]
        ff_pt = ff_beacon[ff_beacon['beiwe'] == pt]
        ff_pt_summary = ff_pt.groupby('date').mean()

        ff_sleep = ff_sleep.append(ff_sleep_pt.merge(ff_pt_summary,left_index=True,right_index=True,how='inner'))
    # cleaning and saving
    ff_sleep.drop(['lat','long','altitude','accuracy',
        'tvoc','lux','no2','co','co2',"pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","temperature_c","rh"],
        axis=1,inplace=True)
    ff_sleep.to_csv(f'{data_dir}data/processed/fitbit_beiwe_beacon-sleep_summary-{study_suffix}.csv')

    return complete_sleep, ff_sleep

def main():
    get_restricted_beacon_datasets(data_dir='../../')
    get_sleep_summaries(data_dir='../../')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
