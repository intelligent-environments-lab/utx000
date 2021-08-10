from datetime import datetime, timedelta

import os
import sys
sys.path.append('../../')
import logging
import math

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from src.analysis import occupancy_detection

import geopy.distance
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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
        fb_daily_sleep = pd.read_csv(f'{self.data_dir}data/processed/fitbit-sleep_daily-{self.study_suffix}.csv',index_col="date",parse_dates=["date","start_time","end_time"],infer_datetime_format=True)
        # sleep stages
        fb_daily_sleep_stages = pd.read_csv(f'{self.data_dir}data/processed/fitbit-sleep_stages_summary-{self.study_suffix}.csv',parse_dates=["start_date","end_date"],infer_datetime_format=True)
        # combining
        fb_all = fb_daily_sleep_stages.merge(fb_daily_sleep,left_on=["end_date","beiwe"],right_on=["date","beiwe"],how="right")
        # filtering data
        fb_all = fb_all[fb_all["main_sleep"] == True]
        # creating features
        for sleep_stage_metric in ["count","minutes"]:
            fb_all[f"nrem_{sleep_stage_metric}"] = fb_all[f"light_{sleep_stage_metric}"] + fb_all[f"deep_{sleep_stage_metric}"] # nrem count and minutes
        
        fb_all[f"rem2nrem"] = fb_all[f"rem_minutes"] / fb_all[f"nrem_minutes"] # ratio of nrem to rem (count and minutes)

        fb_all["tst_fb"] = fb_all["minutes_asleep"] / 60
        for stage in ["rem","nrem","light","deep"]:
            fb_all[f"{stage}_percent"] = fb_all[f"{stage}_minutes"] / (fb_all["tst_fb"]*60)
        fb_all["waso"] = fb_all["minutes_awake"] - fb_all["sol"] - fb_all["wol"]
        fb_all["sol_fb"] = fb_all["sol"]
        fb_all["wol_fb"] = fb_all["wol"]
        fb_all.drop(["sol","wol"],axis="columns",inplace=True)
        # dropping unecessary columns
        fb_all.drop(["main_sleep","duration_ms","minutes_after_wakeup","time_in_bed","minutes_asleep","minutes_to_sleep","minutes_awake"],axis="columns",inplace=True)
        # creating start and end dates for certain participants
        for t in ["start","end"]:
            fb_all[f"{t}_date"] = fb_all.apply(
                    lambda row: row[f'{t}_time'].date() if pd.isnull(row[f'{t}_date']) else row[f'{t}_date'],
                    axis=1)
        # saving and returning
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
                                'wake_count', 'wake_minutes', "beiwe", 'efficiency', 'end_time', 'start_time', "redcap", "beacon",
                                "nrem_count", "nrem_minutes", "rem2nrem", "tst_fb","rem_percent","nrem_percent","light_percent","deep_percent","waso", "sol_fb", "wol_fb",
                                'tst_ema', 'sol_ema', 'naw_ema', 'restful_ema',]
                                
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
            ff_pt_summary = ff_pt.groupby('date').mean().reset_index()
            ff_pt_summary["date"] = pd.to_datetime(ff_pt_summary["date"])

            ff_sleep = ff_sleep.append(ff_sleep_pt.merge(ff_pt_summary,on=["date","beacon","redcap"],how='inner'))
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

    def get_beacon_home_and_asleep(self):
        """
        Gets various statistics for the beacon data when the participant is considered home and asleep
        """
        # Beacon Data During Sleep
        # ------------------------
        print("\tGetting Beacon Data During Fitbit Sleep with GPS and/or CO2 Confirmation")
        data = pd.read_csv(f"{self.data_dir}data/processed/beacon_by_night-{self.study_suffix}.csv",index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)

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

            summarized_df.to_csv(f"{self.data_dir}data/processed/beacon_by_night-summary-{self.study_suffix}.csv")

    def get_beacon_gps_intersection(self, byid="beiwe", join_col="timestamp"):
        """Creates and save intersection between gps and iaq data"""
        # data import
        gps = pd.read_csv("../data/processed/beiwe-gps-ux_s20.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        iaq = pd.read_csv("../data/processed/beacon-ux_s20.csv",parse_dates=["timestamp"],infer_datetime_format=True)

        # resampling gps
        gps_resampled = pd.DataFrame()
        for pt in gps["beiwe"].unique():
            gps_by_pt = gps[gps["beiwe"] == pt]
            gps_by_pt.set_index("timestamp",inplace=True)
            gps_by_pt = gps_by_pt.resample('2T').mean()
            gps_by_pt.reset_index(inplace=True)
            gps_by_pt.dropna(inplace=True)
            gps_by_pt["beiwe"] = pt
            gps_resampled = gps_resampled.append(gps_by_pt)

        # merging dataframes
        merged = gps_resampled.merge(right=iaq,on=[byid,join_col],how="inner")

        # saving
        merged.to_csv(f"{self.data_dir}data/processed/beacon_beiwe-ieq_gps_intersection-{self.study_suffix}.csv")

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

    def get_redcap_ee_survey_summary(self):
        """

        """
        print("\tCombining Environment and Sleep Data")
        # data import
        ee = pd.read_csv(f"{self.data_dir}data/processed/redcap-ee_survey-{self.study_suffix}.csv")
        # merging information
        ee_and_fb = self.fb_sleep_summary.merge(right=ee,on=["beiwe","redcap","beacon"],how="left").dropna(subset=["building_type"])
        # saving
        ee_and_fb.to_csv(f"{self.data_dir}data/processed/redcap_fitbit-environment_and_sleep-{self.study_suffix}.csv",index=False)

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
    sleep = sleep[sleep["main_sleep"] == True] # removing naps and other non-main events
    end_dates = []
    for d in sleep['end_time']:
        end_dates.append(d.date())
    sleep['end_date'] = end_dates
    sleep['end_date'] = pd.to_datetime(sleep["end_time"].dt.date) # should work
    # EMA data
    ema = pd.read_csv(f'{data_dir}data/processed/beiwe-morning_ema-{study_suffix}.csv',
                  index_col=0,parse_dates=True,infer_datetime_format=True)
    # gps data
    gps = pd.read_csv(f'{data_dir}data/processed/beiwe-gps-{study_suffix}.csv',
                 index_col=0,parse_dates=[0,1],infer_datetime_format=True)
    # participant info data for beacon users
    info = pd.read_excel(f'{data_dir}data/raw/utx000/admin/id_crossover.xlsx',sheet_name='beacon',
                    parse_dates=[3,4,5,6])

    beacon_nightly = pd.DataFrame() # df restricted by fitbit and gps
    for pt in sleep['beiwe'].unique():
        if pt in info['beiwe'].values: # only want beacon particiapnts
            # getting data per participant
            gps_pt = gps[gps['beiwe'] == pt]
            sleep_pt = sleep[sleep['beiwe'] == pt]
            beacon_pt = beacon[beacon['beiwe'] == pt]
            info_pt = info[info['beiwe'] == pt]
            lat_pt1 = info_pt['lat'].values[0]
            long_pt1 = info_pt['long'].values[0]
            lat_pt2 = info_pt['lat2'].values[0]
            long_pt2 = info_pt['long2'].values[0]
            ema_pt = ema[ema['beiwe'] == pt]
            # looping through sleep start and end times
            for start_time, end_time in zip(sleep_pt['start_time'],sleep_pt['end_time']):
                if len(beacon_pt[start_time:end_time]) > 0:
                    beacon_pt_night = beacon_pt[start_time:end_time]
                    beacon_pt_night['start_time'] = start_time
                    beacon_pt_night['end_time'] = end_time
                    beacon_pt_night['beiwe'] = pt
                    # gps flag
                    gps_pt_night = gps_pt[start_time:end_time]
                    if len(gps_pt_night) > 0:
                        coords_add_1 = (lat_pt1, long_pt1)
                        coords_add_2 = (lat_pt2, long_pt2)
                        coords_beiwe = (np.nanmean(gps_pt_night['lat']), np.nanmean(gps_pt_night['long']))
                        d1 = geopy.distance.distance(coords_add_1, coords_beiwe).m
                        try:
                            d2 = geopy.distance.distance(coords_add_2, coords_beiwe).m
                        except ValueError:
                            d2 = radius + 1 # dummy value greater than radius
                        if d1 < radius or d2 < radius:
                            beacon_pt_night["home"] = 1
                        else:
                            beacon_pt_night["home"] = 0
                    else:
                        beacon_pt_night["home"] = -1
                    # co2 increasing flag
                    beacon_increase = beacon_pt_night.copy()
                    beacon_increase["sma_co2"] = beacon_increase["co2"].rolling(window=60,center=True,min_periods=30).mean()
                    beacon_increase["dC"] = beacon_increase[f"sma_co2"] - beacon_increase[f"sma_co2"].shift(1) # getting dC
                    beacon_increase["sma_dC"] = beacon_increase["dC"].rolling(window=5,min_periods=3).mean() # getting moving average of increases
                    inc = []
                    for value in beacon_increase["sma_dC"]:
                        if math.isnan(value):
                            inc.append(np.nan)
                        elif value > 0:
                            inc.append(1)
                        else:
                            inc.append(0)
                    beacon_pt_night["increasing_co2"] = np.nanmean(inc)
                    # ema flag
                    beacon_pt_night["ema"] = 1 if end_time.date() in ema_pt.index.date else 0

                    beacon_nightly = beacon_nightly.append(beacon_pt_night)

    beacon_nightly.to_csv(f'{data_dir}data/processed/beacon_by_night-bad_nights_included-{study_suffix}.csv')
    # filtering out manually inspected nights
    nights_to_exclude = pd.read_csv(f"{data_dir}data/interim/bad_analysis_nights.csv")
    for column in nights_to_exclude.columns:
        nights_to_exclude[column] = pd.to_datetime(nights_to_exclude[column])
        
    data_to_save = pd.DataFrame()
    for pt in beacon_nightly["beiwe"].unique():
        bb_pt = beacon_nightly[beacon_nightly["beiwe"] == pt]
        exclude_pt = nights_to_exclude.loc[:,pt]
        if len(exclude_pt) != 0:
            bb_pt = bb_pt[~bb_pt["start_time"].isin(exclude_pt)]
            
        data_to_save = data_to_save.append(bb_pt)
        
    # Data filtered by fitbit nights only
    data_to_save.to_csv(f'{data_dir}data/processed/beacon_by_night-{study_suffix}.csv')

    return data_to_save
            
def main():
    get_restricted_beacon_datasets(data_dir='../../')

    fs = fitbit_sleep(data_dir='../../')
    fs.get_beiwe_summaries()
    fs.get_fitbit_summaries()
    fs.get_beacon_home_and_asleep()
    fs.get_complete_summary()
    fs.get_redcap_ee_survey_summary()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
