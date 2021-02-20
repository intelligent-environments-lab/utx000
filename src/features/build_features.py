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

class fitbit_sleep():
    """
    Class used generate data files that include features from all modalities where the response is Fitbit-measured sleep metrics
    """
    
    def __init__(self,data_dir='../../',study_suffix="ux_s20"):
        self.data_dir = data_dir
        self.study_suffix = study_suffix
        self.fb_sleep_summary = self.get_fitbit_sleep_summary()
        for col in ["start_date","end_date","start_time","end_time"]:
            self.fb_sleep_summary[col] = pd.to_datetime(self.fb_sleep_summary[col])

    # Base Data
    # ---------
    def get_fitbit_sleep_summary(self):
        '''
        Gets summary sleep data from Fitbit including sleep stages and daily sleep summaries

        Returns a dataframes pertaining to the sleep metrics from Fitbit
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
        fb_all = fb_all[fb_all["main_sleep"] == True]
        fb_all.drop(["main_sleep"],axis="columns",inplace=True)
        fb_all.to_csv(f"{self.data_dir}data/processed/fitbit-sleep_summary-{self.study_suffix}.csv",index=False)

        return fb_all

    # Helper Functions
    # ----------------
    def get_ema_distribution_timestamp(self, row):
        """
        Returns the evening EMA timestamp prior to the night of sleep 
        """
        if row["start_time"].hour < 19:
            # subtracting a day if the participant went to bed after midnight (prior to time of EMA distribution)
            d = row["start_time"] - timedelta(days=1)
        else:
            d = row["start_time"]
            
        return datetime(d.year,d.month,d.day,19,0,0)

    def insert_ema_timestamp(self, row, ema):
        """
        Inserts participant EMA submission time if between the ema distribution time and sleep start time, NaN otherwise
        """

        bid = row["beiwe"]
        ema_by_id = ema[ema["beiwe"] == bid]
        for ts in ema_by_id["timestamp"]:
            if ts.date == row["start_time"].date and ts.hour < row["start_time"].hour:
                return ts
            elif ts > row["ema_evening_distribution"] and ts < row["start_time"]:
                return ts
            
        return np.nan

    # Feature-Building Functions
    # --------------------------
    def get_beiwe_summaries(self):
        """
        Gets the various summaries of Beiwe metrics
        """
        # Morning EMA
        # -----------
        print('\tGetting Data for Fitbit and EMAs')
        # Defining Helper Function
        def fix_ema_timestamps(ema,fb,verbose=False):
            """
            Corrects the EMA timestamps based on nearby Fitbit sleep data

            Inputs:
            - ema: dataframe with the EMA data
            - fb: dataframe with the fitbit sleep data
            - verbose: boolean to show output or not

            Returns a of dataframe of the EMA data with a revised date column
            """
            ema_with_dates = pd.DataFrame()
            for pt in ema["beiwe"].unique():
                dates = []
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

                # adding dates and appending to final dataframe
                ema_pt["date"] = dates
                ema_with_dates = ema_with_dates.append(ema_pt)

            return ema_with_dates

        # Self-report/EMA sleep
        ema_sleep = pd.read_csv(f'{self.data_dir}data/processed/beiwe-morning_ema-{self.study_suffix}.csv',index_col=0,parse_dates=True,infer_datetime_format=True)
        for column in ['tst','sol','naw','restful']:
            ema_sleep = ema_sleep[ema_sleep[column].notna()]
        ema_sleep = fix_ema_timestamps(ema_sleep,self.fb_sleep_summary)
        ema_sleep["date"] = pd.to_datetime(ema_sleep["date"])

        # Getting complete sleep dataframe
        complete_sleep = pd.DataFrame() # dataframe to append to
        pt_list = np.intersect1d(self.fb_sleep_summary['beiwe'].unique(),ema_sleep['beiwe'].unique())
        for pt in pt_list:
            ema_sleep_beiwe = ema_sleep[ema_sleep['beiwe'] == pt]
            fb_beiwe = self.fb_sleep_summary[self.fb_sleep_summary['beiwe'] == pt]
            complete_sleep = complete_sleep.append(fb_beiwe.merge(ema_sleep_beiwe,left_on=['end_date',"beiwe","redcap","beacon"],right_on=['date',"beiwe","redcap","beacon"],how='inner'))

        # renaming and dropping for easier use
        complete_sleep.set_index('date',inplace=True)
        complete_sleep.drop(['content', 'stress', 'lonely', 'sad', 'energy'],axis=1,inplace=True)
        complete_sleep.columns = ['start_date', 'end_date', 'deep_count', 'deep_minutes',
                                'light_count', 'light_minutes', 'rem_count', 'rem_minutes',
                                'wake_count', 'wake_minutes', "beiwe",'tst_fb', 'efficiency', 'end_time',
                                'minutes_after_wakeup', 'minutes_asleep', 'minutes_awake',
                                'minutes_to_sleep', 'start_time', 'time_in_bed', "redcap","beacon",
                                'tst_ema', 'sol_ema', 'naw_ema', 'restful_ema',]
        complete_sleep["tst_fb"] /= 3600000
        complete_sleep.to_csv(f'{self.data_dir}data/processed/beiwe_fitbit-sleep_summary-{self.study_suffix}.csv')

        # Fully Filtered
        # --------------
        print('\tGetting Data for Fitbit and EMAs Filtered by GPS and Beacon Measurements')
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
        ff_sleep.to_csv(f'{self.data_dir}data/processed/beiwe_fitbit-beacon_and_gps_filtered_sleep_summary-{self.study_suffix}.csv')

        # Evening Mood
        # ------------
        print('\tGetting Data for Fitbit and Evening Mood Reports')
        # importing evening emas
        ema_evening = pd.read_csv(f"{self.data_dir}data/processed/beiwe-evening_ema-{self.study_suffix}.csv",parse_dates=["timestamp"])
        # importing fitbit sleep metrics
        fb_all_sleep = pd.read_csv(f"{self.data_dir}data/processed/fitbit-sleep_summary-{self.study_suffix}.csv",parse_dates=["start_date","end_date","start_time","end_time"])
        # adding in ema distribution time prior to sleep
        fb_all_sleep["ema_evening_distribution"] = fb_all_sleep.apply(self.get_ema_distribution_timestamp, axis="columns")
        # checking to see if ema was submitted between distribution time and sleep start time
        fb_all_sleep["ema_evening_timestamp"] = fb_all_sleep.apply(lambda x: self.insert_ema_timestamp(x,ema_evening), axis="columns")
        # combining on matching evening timestamps (and ids)
        evening_ema_and_sleep = fb_all_sleep.merge(right=ema_evening,left_on=["beiwe","redcap","beacon","ema_evening_timestamp"],right_on=["beiwe","redcap","beacon","timestamp"])
        evening_ema_and_sleep.drop(["ema_evening_timestamp","ema_evening_distribution"],axis="columns",inplace=True)
        evening_ema_and_sleep.to_csv(f"{self.data_dir}data/processed/beiwe_fitbit-evening_mood_and_sleep-{self.study_suffix}.csv",index=False)

        # Daily Mood
        # ----------
        print('\tGetting Data for Fitbit and Daily-Averaged Mood Reports')
        # importing, combining, and saving "daily" EMA data
        morning = pd.read_csv(f"{self.data_dir}data/processed/beiwe-morning_ema-{self.study_suffix}.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        morning["date"] = morning["timestamp"].dt.date
        evening = pd.read_csv(f"{self.data_dir}data/processed/beiwe-evening_ema-{self.study_suffix}.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        evening["date"] = evening["timestamp"].dt.date
        emas = morning.merge(evening,left_on=["date","beiwe","redcap","beacon"],right_on=["date","beiwe","redcap","beacon"],suffixes=('_morning', '_evening'))
        for mood in ["content","stress","lonely","sad","energy"]:
            emas[f"{mood}_mean"] = emas[[f"{mood}_morning",f"{mood}_evening"]].mean(axis=1)
        emas["date"] = pd.to_datetime(emas["date"])
        emas.to_csv(f"{self.data_dir}data/processed/beiwe-daily_ema-{self.study_suffix}.csv",index=False)
        # importing fitbit data and combining fitbit data with ema data
        fb_mood = self.fb_sleep_summary.merge(emas,left_on=["end_date","beiwe","redcap","beacon"],right_on=["date","beiwe","redcap","beacon"])
        fb_mood.drop(['tst', 'sol', 'naw', 'restful', 'date'],axis=1,inplace=True)
        fb_mood.to_csv(f"{self.data_dir}data/processed/beiwe_fitbit-daily_mood_and_sleep-{self.study_suffix}.csv",index=False)

    def get_fitbit_summaries(self):
        """
        Gets the various summaries of Fitbit metrics
        """

        # Activity Data Prior to Sleep Event
        # ----------------------------------
        print('\tGetting Data for Fitbit Sleep and Activity Data')
        # importing activity data summarized by day
        activity = pd.read_csv(f"{self.data_dir}data/processed/fitbit-daily-{self.study_suffix}.csv",parse_dates=["timestamp"])
        # adding date column based on the date of the end time minus one day to account for people sleeping after midnight
        fb_all_sleep = self.fb_sleep_summary.copy()
        fb_all_sleep["date"] = pd.to_datetime(fb_all_sleep["end_time"].dt.date - timedelta(days=1))
        # combining data and saving
        activity_and_sleep = fb_all_sleep.merge(right=activity,left_on=["date","beiwe"],right_on=["timestamp","beiwe"])
        activity_and_sleep.to_csv(f"{self.data_dir}data/processed/fitbit_fitbit-daily_activity_and_sleep-{self.study_suffix}.csv",index=False)

    def get_beacon_summaries(self):
        """
        Gets various statistics for the beacon data when the participant is considered home and asleep
        """
        # Beacon Data During Sleep
        # ------------------------
        print("\tGetting Beacon Data During Fitbit Sleep with GPS Confirmation")
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
        beacon_fitbit_summary = self.fb_sleep_summary.merge(summarized_df,left_on=["start_time","end_time","beiwe","beacon","redcap"],right_on=["start_time","end_time","beiwe","beacon","redcap"])
        beacon_fitbit_summary.to_csv(f"{self.data_dir}data/processed/beacon_fitbit-ieq_and_sleep-{self.study_suffix}.csv",index=False)

    def get_complete_summary(self):
        """
        Compiles all possible data modalities and sleep that overlap
        """
        # Data Import
        # -----------
        # summarized beacon
        beacon_summary = pd.read_csv(f"{self.data_dir}data/processed/beacon-fb_and_gps_filtered_summary-{self.study_suffix}.csv",parse_dates=["start_time","end_time"])
        # morning ema
        ema_morning = pd.read_csv(f"{self.data_dir}data/processed/beiwe-morning_ema-{self.study_suffix}.csv",parse_dates=["timestamp"])
        # evening ema
        ema_evening = pd.read_csv(f"{self.data_dir}data/processed/beiwe-evening_ema-{self.study_suffix}.csv",parse_dates=["timestamp"])
        # fitbit activity
        activity = pd.read_csv(f"{self.data_dir}data/processed/fitbit-daily-{self.study_suffix}.csv",parse_dates=["timestamp"])

        # Combining One Step at a Time
        # ----------------------------
        # beacon and morning ema
        # simply merging on end/wake day of sleep and ema submission
        beacon_summary["date"] = beacon_summary["end_time"].dt.date
        ema_morning["date"] = ema_morning["timestamp"].dt.date
        # renaming columns for later when we add in evening ema data
        ema_morning.rename({"timestamp":"timestamp_ema_morning","content":"content_morning","stress":"stress_morning","lonely":"lonely_morning","sad":"sad_morning","energy":"energy_morning"},axis=1,inplace=True)
        combined = beacon_summary.merge(right=ema_morning,left_on=["beiwe","redcap","beacon","date"],right_on=["beiwe","redcap","beacon","date"]) # merging on matching columns
        
        # beacon and both emas
        # the most complicated merge - merge if ema was submitted prior to sleep that evening
        ema_evening.rename({"content":"content_evening","stress":"stress_evening","lonely":"lonely_evening","sad":"sad_evening","energy":"energy_evening"},axis=1,inplace=True) # renamed
        # adding in ema distribution time prior to sleep
        combined["ema_evening_distribution"] = combined.apply(self.get_ema_distribution_timestamp, axis="columns")
        # checking to see if ema was submitted between distribution time and sleep start time
        combined["ema_evening_timestamp"] = combined.apply(lambda x: self.insert_ema_timestamp(x,ema_evening), axis="columns")
        # combining on matching evening timestamps (and ids)
        more_combined = combined.merge(right=ema_evening,left_on=["beiwe","redcap","beacon","ema_evening_timestamp"],right_on=["beiwe","redcap","beacon","timestamp"])
        more_combined.drop(["timestamp","date","ema_evening_distribution"],axis="columns",inplace=True) # dropping helper columns
        
        # beacon, both emas, and activity
        # mergining on the activity date and the sleep end date minus a day to account for people falling asleep after midnight
        more_combined["date"] = pd.to_datetime(more_combined["end_time"].dt.date - timedelta(days=1))
        more_more_combined = more_combined.merge(right=activity,left_on=["date","beiwe"],right_on=["timestamp","beiwe"])

        # adding in Fitbit sleep data
        # the final merge - since beacon summary data are created from the Fitbit sleep data, we can merge on similarly named columns that already exist
        final_combined = more_more_combined.merge(right=self.fb_sleep_summary,left_on=["start_time","end_time","beiwe","redcap","beacon"],right_on=["start_time","end_time","beiwe","redcap","beacon"])
        # dropping helper, repeated, and unecessary columns for a tidier dataframe and saving
        final_combined.drop(["date","timestamp","start_date","end_date","bmi","bmr","fat","weight","food_calories_logged","water_logged"],axis="columns",inplace=True)
        final_combined.to_csv(f"{self.data_dir}data/processed/all_modalities-fb_and_gps_filtered-{self.study_suffix}.csv",index=False)

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
    #get_restricted_beacon_datasets(data_dir='../../')

    fs = fitbit_sleep(data_dir='../../')
    fs.get_beiwe_summaries()
    fs.get_fitbit_summaries()
    fs.get_beacon_summaries()
    fs.get_complete_summary()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
