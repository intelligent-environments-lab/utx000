from datetime import datetime, timedelta

import os
import sys
import logging

import pandas as pd
import numpy as np

import geopy.distance

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

class nightly_summaries():
    """
    Class used to summarize data by night when the participant is assumed to be asleep
    """
    
    def __init__(self,data_dir='../../',study_suffix="ux_s20"):
        self.data_dir = data_dir
        self.study_suffix = study_suffix

    def get_sleep_summaries(self):
        '''
        Gets summary sleep data from Fitbit, EMAs and Fitbit, and EMAs and Fitbit from Participants with beacons and when they are home

        Inputs:
        - data_dir: string corresponding to the location of the "data" dir
        - study_suffix: string used to find the file and save the new files

        Returns two dataframes pertaining to the sleep data for both datasets
        - all fitbit: sleep summaries and stage summaries from Fitbit
        - complete: all nights with both EMA and Fitbit recordings
        - fully filtered: nights when participants are home and have beacon data too
        '''
        # Fitbit Summary and Sleep Stages Summary
        # ---------------------------------------
        print("\tCombining Fitbit Sleep Measurements")
        # sleep summary
        fb_daily_sleep = pd.read_csv(f'{self.data_dir}data/processed/fitbit-sleep_daily-{self.study_suffix}.csv',index_col="date",parse_dates=True,infer_datetime_format=True)
        # sleep stages
        fb_daily_sleep_stages = pd.read_csv(f'{self.data_dir}data/processed/fitbit-sleep_stages_summary-{self.study_suffix}.csv',parse_dates=["start_date","end_date"],infer_datetime_format=True)
        # combining and saving
        fb_all = fb_daily_sleep_stages.merge(fb_daily_sleep,left_on=["end_date","beiwe"],right_on=["date","beiwe"],how="inner")
        fb_all.to_csv(f"{self.data_dir}data/processed/fitbit-sleep_data_summary-{self.study_suffix}.csv",index=False)

        # EMA and Fitbit
        # --------------
        print('\tGetting Combined Sleep Summary from Fitbit and EMAs')

        def fix_ema_timestamps(ema,fb,verbose=False):
            """
            Corrects the EMA timestamps based on nearby Fitbit sleep data

            Inputs:
            - ema: dataframe with the EMA data
            - fb: dataframe with the fitbit sleep data
            - verbose: boolean to show output or not

            Returns a of dataframe of the EMA data with a revised date column
            """
            dates = []
            for pt in ema["beiwe"].unique():
                ema_pt = ema[ema['beiwe'] == pt]
                try:
                    fb_pt = fb[fb['beiwe'] == pt]
                except KeyError:
                    continue
                for ema_dt in ema_pt.index:
                    dates.append(ema_dt.date())
                    if ema_dt.hour < 9:
                        try:
                            fb_info = fb_pt.loc[ema_dt.date(),:]
                        except KeyError:
                            fb_info = 0
                            if verbose:
                                print(f"EMA Hour:\t{ema_dt.hour}\nNo Fitbit data for this day\n")
                            #dates.append(ema_dt.date())
                        if type(fb_info) != int:
                            fb_start = pd.to_datetime(fb_info["start_time"])
                            fb_end = pd.to_datetime(fb_info["end_time"])
                            if ema_dt.day == fb_start.day:
                                if ema_dt.hour < fb_start.hour:
                                    if verbose:
                                        print(f"EMA DT:\t{ema_dt}\nFB DT:\t{fb_start}\nSubtract an Hour\n")
                                    dates[-1] -= timedelta(days=1)
                            elif ema_dt.day == fb_end.day:
                                if ema_dt.hour > fb_end.hour:
                                    dow = ema_dt.strftime("%A")
                                    if verbose:
                                        print(f"EMA DT:\t{ema_dt}\nFB DT:\t{fb_end}\nNo Change - {dow}\n")
                            else:
                                if verbose:
                                    print(f"EMA DT:\t{ema_dt}\nFB SDT:\t{fb_start}\nFB EDT:\t{fb_end}\nSomething isn\'t right here")
            if verbose:
                print(f"Number of Surveys:\t{len(ema)}\nNumber of Dates:\t{len(dates)}")
                
            # adding date column and returing
            ema["date"] = dates
            return ema

        # Self-report/EMA sleep
        ema_sleep = pd.read_csv(f'{self.data_dir}data/processed/beiwe-morning_ema-{self.study_suffix}.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
        for column in ['tst','sol','naw','restful']:
            ema_sleep = ema_sleep[ema_sleep[column].notna()]
        ema_sleep = fix_ema_timestamps(ema_sleep,fb_all)
        ema_sleep["date"] = pd.to_datetime(ema_sleep["date"])

        # Getting complete sleep dataframe
        complete_sleep = pd.DataFrame() # dataframe to append to
        pt_list = np.intersect1d(fb_daily_sleep['beiwe'].unique(),ema_sleep['beiwe'].unique())
        for pt in pt_list:
            ema_sleep_beiwe = ema_sleep[ema_sleep['beiwe'] == pt]
            fb_beiwe = fb_all[fb_all['beiwe'] == pt]
            complete_sleep = complete_sleep.append(fb_beiwe.merge(ema_sleep_beiwe,left_on='end_date',right_on='date',how='inner'))
        # Saving
        complete_sleep.set_index('date',inplace=True)
        complete_sleep["beiwe"] = complete_sleep["beiwe_x"]
        complete_sleep.drop(["beiwe_x","beiwe_y"],axis=1,inplace=True)
        complete_sleep.to_csv(f'{self.data_dir}data/processed/fitbit_beiwe-sleep_summary-{self.study_suffix}.csv')

        # Fully Filtered
        # --------------
        print('\tGetting Sleep Summary for Fully Filtered Beacon Dataset')
        # Fully filtered beacon dataset to cross-reference
        ff_beacon = pd.read_csv(f'{self.data_dir}data/processed/beacon-fb_ema_and_gps_filtered-{self.study_suffix}.csv',
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
        ff_sleep.to_csv(f'{self.data_dir}data/processed/fitbit_beiwe_beacon-sleep_summary-{self.study_suffix}.csv')

        return fb_all,complete_sleep, ff_sleep

    def get_fitbit_summaries(self):
        """
        Gets the various summaries of Fitbit metrics whent the participant is considered home and asleep
        """

        pass

    def get_beacon_summaries(self):
        """
        Gets various statistics for the beacon data when the participant is considered home and asleep
        """

        data = pd.read_csv(f"{self.data_dir}data/processed/beacon-fb_and_gps_filtered-{self.study_suffix}.csv",index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)
        summarized_df = pd.DataFrame()
        for s in ["mean","median","delta","delta_percent"]:
            beacon_by_s = pd.DataFrame()
            for pt in data["beiwe"].unique():
                data_by_pt = data[data["beiwe"] == pt]
                ids = data_by_pt[["end_time","beacon","beiwe","fitbit","redcap"]]
                data_by_pt.drop(["end_time","beacon","beiwe","fitbit","redcap"],axis=1,inplace=True)
                if s == "mean":
                    data_s_by_pt = data_by_pt.groupby("start_time").mean()
                elif s == "median":
                    data_s_by_pt = data_by_pt.groupby("start_time").median()
                elif s == "delta":
                    little = data_by_pt.groupby("start_time").min()
                    big = data_by_pt.groupby("start_time").max()
                    data_s_by_pt = big - little
                else:
                    little = data_by_pt.groupby("start_time").min()
                    big = data_by_pt.groupby("start_time").max()
                    data_s_by_pt = (big - little) / little * 100

                data_s_by_pt = data_s_by_pt.add_suffix(f"_{s}")
                data_s_by_pt["end_time"] = ids["end_time"].unique()
                for col in ids.columns[1:]:
                    data_s_by_pt[col] = ids[col][0]

                beacon_by_s = beacon_by_s.append(data_s_by_pt)

            if len(summarized_df) == 0:
                summarized_df = beacon_by_s
            else:
                summarized_df = summarized_df.merge(beacon_by_s,on=["start_time","end_time","beacon","beiwe","fitbit","redcap"])

        summarized_df.to_csv(f"{self.data_dir}data/processed/beacon-fb_and_gps_filtered_summary-{self.study_suffix}.csv")
        # combining fitbit data to the beacon summary
        fb_all = pd.read_csv(f"{self.data_dir}data/processed/fitbit-sleep_data_summary-{self.study_suffix}.csv",parse_dates=["start_time","end_time"],infer_datetime_format=True)
        beacon_fitbit_summary = fb_all.merge(summarized_df,left_on=["start_time","end_time","beiwe"],right_on=["start_time","end_time","beiwe"])
        beacon_fitbit_summary.to_csv(f"{self.data_dir}data/processed/beacon_fitbit-sleep_data_summary-{self.study_suffix}.csv",index=False)
        return summarized_df, beacon_fitbit_summary

class daily_summaries():
    """
    Class meant to summarize data prior to the participant sleeping
    """

    def __init__(self,data_dir='../../',study_suffix="ux_s20"):
        self.data_dir = data_dir
        self.study_suffix = study_suffix
        # importing all possible fitbit sleep data that we will use to filter the other datasets
        self.fb_sleep = pd.read_csv(f"{self.data_dir}data/processed/fitbit-sleep_data_summary-{self.study_suffix}.csv",parse_dates=["end_date"])

    def get_ema_mood_summaries(self):
        """
        Summarizes the mood scores from EMAs submitted on the same day
        """
        # importing, combining, and saving "daily" EMA data
        morning = pd.read_csv(f"{self.data_dir}data/processed/beiwe-morning_ema-ux_s20.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        morning["date"] = morning["timestamp"].dt.date
        evening = pd.read_csv(f"{self.data_dir}data/processed/beiwe-evening_ema-ux_s20.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        evening["date"] = evening["timestamp"].dt.date
        emas = morning.merge(evening,left_on=["date","beiwe"],right_on=["date","beiwe"],suffixes=('_morning', '_evening'))
        emas["date"] = pd.to_datetime(emas["date"])
        emas.to_csv(f"{self.data_dir}data/processed/beiwe-daily_ema-{self.study_suffix}")
        # importing fitbit data and combining fitbit data with ema data
        fb_mood = self.fb_sleep.merge(emas,left_on=["end_date","beiwe"],right_on=["date","beiwe"])
        fb_mood.drop(['tst', 'sol', 'naw', 'restful', 'date'],axis=1,inplace=True)
        fb_mood.to_csv(f"{self.data_dir}data/processed/fitbit_beiwe-sleep_and_mood-{self.study_suffix}.csv",index=False)

    def get_fitbit_summaries(self):
        """
        Summarizes fitbit-related metrics and filters the data down to only include data with fitbit-measured sleep
        """
        # importing non-sleep fitbit data, combining, and saving
        fb_activity = pd.read_csv(f"{self.data_dir}data/processed/fitbit-daily-{self.study_suffix}.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        fb_all = self.fb_sleep.merge(fb_activity,left_on=["end_date","beiwe"],right_on=["timestamp","beiwe"])
        fb_all.to_csv(f"{self.data_dir}data/processed/fitbit-sleep_activity_daily-{self.study_suffix}.csv",index=False)

        return fb_all


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

def main():
    get_restricted_beacon_datasets(data_dir='../../')

    ns = nightly_summaries(data_dir='../../')
    ns.get_sleep_summaries()
    ns.get_beacon_summaries()
    ns.get_fitbit_summaries()

    ds = daily_summaries(data_dir='../../')
    ds.get_ema_mood_summaries()
    ds.get_fitbit_summaries()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
