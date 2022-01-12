import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

class PreProcess:

    def __init__(self) -> None:
        pass

class Classify:

    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data") -> None:
        """
        Initializing method

        Parameters
        ----------
        study : str, default "utx000"
            study name
        study_suffix : str, default "ux_s20"
            study suffix ID
        data_dir : str, default "../../data"
            path to the "data" directory within the project
        """
        self.study = study
        self.suffix = study_suffix
        self.data_dir = data_dir

        # Loading Data
        # ------------
        ## Beacon
        self.beacon_all = pd.read_csv(f"{data_dir}/processed/beacon-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        self.beacon_all.drop(["redcap","fitbit"],axis=1,inplace=True)
        self.beacon_all.dropna(subset=["co2"],inplace=True)

        self.beacon_nightly = pd.read_csv(f"{self.data_dir}/processed/beacon_by_night-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)
        self.beacon_nightly.drop(["no2","increasing_co2","ema","redcap","fitbit"],axis=1,inplace=True)
        self.beacon_nightly.dropna(subset=["co2"],inplace=True)

        ## GPS
        self.gps = pd.read_csv(f"{self.data_dir}/processed/beiwe-gps_beacon_pts-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        
    def add_label(self,home_labels=[1],verbose=False):
        """
        Includes a column that corresponds to the occupied label 

        Parameters
        ----------
        home_labels : list of int, default [1]
            labels to use that indicate if the participant is home in [0,1]
            0 indicates participants were confirmed home by sleep episode and CO2/T measurements
            1 indicates participants were confirmed home by GPS/address cross-reference
        verbose : boolean, default False
            increased output for debugging purposes
        
        Returns
        -------

        """
        # occupied label
        beacon_occupied = self.beacon_nightly.copy() # nightly measurements are from occupied periods
        beacon_occupied = beacon_occupied[beacon_occupied["home"].isin(home_labels)] # ensures we use gps-confirmed data
        beacon_occupied["label"] = "occupied" # add label
        #data = self.beacon_all.merge(right=occupied_data[["label","beiwe"]],on=["timestamp","beiwe"],how="left")

        # unoccupied label
        beacon_unoccupied = pd.DataFrame()
        # looping through each participant because we lose ID information when we `groupby`
        for pt in self.beacon_nightly["beiwe"].unique():
            # participant-specific data
            beacon_night_pt = self.beacon_nightly[self.beacon_nightly["beiwe"] == pt]
            beacon_all_pt = self.beacon_all[self.beacon_all["beiwe"] == pt].reset_index()
            gps_pt = self.gps[self.gps["beiwe"] == pt]
            occupied_pt = pd.DataFrame()
            
            # looping through sleep episodes to get gps data from occupied periods
            for s, e in zip(beacon_night_pt["start_time"].unique(),beacon_night_pt["end_time"].unique()):
                occupied_pt = occupied_pt.append(gps_pt.loc[s:e])
            
            # merging and pulling out non-overlapping data
            unoccupied_pt = gps_pt.reset_index().merge(right=occupied_pt.reset_index()[["beiwe","timestamp"]],on=["beiwe","timestamp"],how="left",indicator=True)
            unoccupied_only = unoccupied_pt[unoccupied_pt["_merge"] == "left_only"]
            # resampling to timestamp consistent with beacon data
            unoccupied_resampled = unoccupied_only.set_index("timestamp").resample("2T").mean().dropna()
            unoccupied_resampled["beiwe"] = pt # adding back in the participant ID
            if verbose:
                print(f"{pt}: {len(unoccupied_resampled)}")
            
            # merging gps data from unoccupied periods with beacon data
            iaq_unoccupied = beacon_all_pt.merge(right=unoccupied_resampled.reset_index(),on=["beiwe","timestamp"],how="inner")
            beacon_unoccupied = beacon_unoccupied.append(iaq_unoccupied)

        # adding labels, combining, and dropping any pesky duplicates
        beacon_occupied["label"] = "occupied"
        beacon_unoccupied["label"] = "unoccupied"
        labeled_data = beacon_occupied.append(beacon_unoccupied.set_index("timestamp"))
        labeled_data.drop_duplicates(subset=["beiwe","co2"],inplace=True)
        labeled_data.reset_index(inplace=True)
        
        # adding both labels to beacon measurements and cleaning
        data = self.beacon_all.reset_index().merge(labeled_data[["beiwe","timestamp","label"]],on=["beiwe","timestamp"],how="inner")
        pts_with_one_label = []
        for pt in data["beiwe"].unique():
            data_pt = data[data["beiwe"] == pt]
            if len(data_pt["label"].unique()) != 2:
                pts_with_one_label.append(pt)
        data = data[~data["beiwe"].isin(pts_with_one_label)]

        self.data = data.set_index("timestamp")

    def resample_data(self, rate=15, by_id="beiwe"):
        """
        Resamples data to the given rate

        Parameters
        ----------
        rate : int, default 15
            resample rate in minutes
        by_id : str, default "beiwe"
            ID to prse out data by

        Returns
        -------
        resampled : DataFrame
            resampled data from df
        """
        resampled = pd.DataFrame()
        # have to parse out data because of duplicate timestamps and because we lose beiwe IDs
        for pt in self.data[by_id].unique():
            data_pt = self.data[self.data[by_id] == pt]
            data_pt.resample(f"{rate}T").mean()
            data_pt[by_id] = pt # adding ID back in
            resampled = resampled.append(data_pt)

        self.data = resampled

    def co2_comparison(self, participants=None, by_id="beiwe", occ_label="occupied", unocc_label="unoccupied"):
        """
        Compares distributions of co2 measurements between occupied and unoccupied conditions
        """

        if participants == None:
            pt_list = self.data[by_id].unique()
        elif isinstance(participants,list):
            pt_list = participants
        else:
            pt_list = [participants] 

        for pt in pt_list:
            _, ax =plt.subplots(figsize=(12,4))
            data_pt = self.data[self.data[by_id] == pt]
            occupied_co2 = data_pt[data_pt["label"] == occ_label]
            unoccupied_co2 = data_pt[data_pt["label"] == unocc_label]
            sns.kdeplot(x="co2",data=occupied_co2,
                lw=2,color="seagreen",cut=0,
                label=occ_label.title(),ax=ax)
            sns.kdeplot(x="co2",data=unoccupied_co2,
                lw=2,color="firebrick",cut=0,
                label=unocc_label.title(),ax=ax)
            # x-axis
            ax.set_xlabel("CO$_2$ Concentration (ppm",fontsize=16)
            # y-axis
            ax.set_ylabel("Density",fontsize=16)
            # remainder
            ax.tick_params(labelsize=12)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            ax.legend(frameon=False,ncol=1,fontsize=14)
            ax.set_title(pt,fontsize=16)

            plt.show()
            plt.close()

class manual_inspection:
    
    def __init__(self,pt,data_dir="../",threshold=0.75):
        self.pt = pt # beiwe id
        self.threshold = threshold
        # beacon data
        complete = pd.read_csv(f"{data_dir}data/processed/beacon-ux_s20.csv", parse_dates=["timestamp"],infer_datetime_format=True)
        filtered = pd.read_csv(f"{data_dir}data/processed/beacon-fb_and_gps_filtered-ux_s20.csv",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)
        self.complete = complete[complete["beiwe"] == self.pt]
        self.filtered = filtered[filtered["beiwe"] == self.pt]
        # fitbit data
        fitbit = pd.read_csv(f"{data_dir}data/processed/fitbit-sleep_summary-ux_s20.csv",parse_dates=["start_time","end_time"],infer_datetime_format=True)
        self.sleep = fitbit[fitbit["beiwe"] == self.pt]
        # gps data
        gps = pd.read_csv(f"{data_dir}data/processed/beiwe-gps-ux_s20.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        self.gps = gps[gps["beiwe"] == self.pt]
        # beacon data derivatives
        self.set_morning_beacon_data()
        self.set_beacon_before_sleep()
        self.set_increasing_periods(self.complete,"co2")
        self.set_increasing_periods(self.filtered,"co2")
        self.set_increasing_only()
        self.set_beacon_by_sleep()
        self.set_beacon_while_occupied()
        self.set_beacon_gps_occupied()
        
    def set_morning_beacon_data(self,time_column="timestamp",num_hours=3):
        """gets the beacon data from the morning"""
        morning_df = pd.DataFrame()
        all_data = self.complete.copy()
        all_data.set_index(time_column,inplace=True)
        for wake_time in self.filtered['end_time'].unique():
            temp = all_data[wake_time:pd.to_datetime(wake_time)+timedelta(hours=num_hours)]
            temp['start_time'] = wake_time
            morning_df = morning_df.append(temp)

        self.morning = morning_df.reset_index()
        
    def set_beacon_before_sleep(self,time_column="timestamp",num_hours=1):
        """sets beacon data prior to sleeping"""
        prior_to_sleep_df = pd.DataFrame()
        all_data = self.complete.copy()
        all_data.set_index(time_column,inplace=True)
        for sleep_time in self.filtered['start_time'].unique():
            temp = all_data[pd.to_datetime(sleep_time)-timedelta(hours=num_hours):pd.to_datetime(sleep_time)+timedelta(hours=1)]
            temp['end_time'] = sleep_time
            prior_to_sleep_df = prior_to_sleep_df.append(temp)

        self.prior = prior_to_sleep_df.reset_index()
        
    def plot_timeseries(self,df,variable,time_column="timestamp",re=False,**kwargs):
        """plots timeseries of the given variable"""
        fig, ax = plt.subplots(figsize=(24,4))
        try:
            if "time_period" in kwargs.keys():
                df = df.set_index(time_column)[kwargs["time_period"][0]:kwargs["time_period"][1]].reset_index()
            # plotting
            ax.scatter(df[time_column],df[variable],color="black",s=10)
            # formatting
            if "event" in kwargs.keys():
                ax.axvline(kwargs["event"],linestyle="dashed",linewidth=3,color="firebrick")
            if "ylim" in kwargs.keys():
                ax.set_ylim(kwargs["ylim"])
                
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)

            if re:
                return ax
            
            plt.show()
            plt.close()
        except Exception as e:
            print(e)
            
    def plot_individual_days(self,dataset,variable="co2",**kwargs):
        """plots the individual days"""
        t = "start_time" if "start_time" in dataset.columns else "end_time"
        print(t)
        for event in dataset[t].unique():
            self.plot_timeseries(dataset[dataset[t] == event],variable,event=pd.to_datetime(event),**kwargs)
            
    def set_increasing_periods(self,dataset,variable,averaging_window=60,increase_window=5,stat="mean",plot=False):
        """finds increasing periods"""
         # smooting data
        if stat == "mean":
            dataset[f"sma_{variable}"] = dataset[variable].rolling(window=averaging_window,center=True,min_periods=int(averaging_window/2)).mean()
        else:
            dataset[f"sma_{variable}"] = dataset[variable].rolling(window=averaging_window,center=True,min_periods=int(averaging_window/2)).median()
        dataset["dC"] = dataset[f"sma_{variable}"] - dataset[f"sma_{variable}"].shift(1) # getting dC
        dataset["sma_dC"] = dataset["dC"].rolling(window=increase_window).mean() # getting moving average of increases
        inc = []
        for value in dataset["sma_dC"]:
            if math.isnan(value):
                inc.append(np.nan)
            elif value > 0:
                inc.append(1)
            else:
                inc.append(0)
        dataset["increasing"] = inc
        #dataset["increasing"] = [1 if value > 0 else 0 for value in dataset["sma_dC"]] # creating column for increasing concentration
        
        if plot:
            fig, ax = plt.subplots(figsize=(24,4))
            ax.scatter(self.complete["timestamp"],self.complete[variable],color="black",s=10,alpha=0.7,zorder=1)
            inc = dataset[dataset["increasing"] == 1]
            ax.scatter(inc["timestamp"],inc[variable],color="seagreen",s=5,zorder=2)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
                
    def set_increasing_only(self):
        """beacon data over increasing periods only"""
        self.inc = self.complete[self.complete["increasing"] == 1]
    
    def set_beacon_by_sleep(self):
        """beacon data during sleep events"""
        beacon_by_fitbit = pd.DataFrame()
        for s, e in zip(self.sleep["start_time"].unique(),self.sleep["end_time"].unique()):
            beacon_temp = self.complete.set_index("timestamp")[pd.to_datetime(s):pd.to_datetime(e)].reset_index()
            beacon_temp["start_time"] = s
            beacon_temp["end_time"] = e
            beacon_by_fitbit = beacon_by_fitbit.append(beacon_temp)
            
        self.beacon_during_sleep = beacon_by_fitbit
        
    def set_beacon_while_occupied(self,**kwargs):
        """beacon data when the bedroom is occupied"""
        beacon_percent = self.beacon_during_sleep.drop(["sma_co2","dC","sma_dC","increasing"],axis="columns").merge(right=self.beacon_during_sleep.groupby("start_time").mean().reset_index()[["increasing","start_time"]],on="start_time",how="left")
        if "threshold" in kwargs.keys():
            self.threshold = kwargs["threshold"]

        self.occupied = beacon_percent[beacon_percent["increasing"] > self.threshold]
        
    def set_beacon_gps_occupied(self):
        """beacon data when occupied or gps confirms home"""
        self.fully_filtered = self.filtered.append(self.occupied).drop_duplicates(subset=["beiwe","timestamp"])
        
    def plot_overlap(self,**kwargs):
        fig, gps_ax = plt.subplots(figsize=(29,6))
        gps_ax.scatter(self.gps["timestamp"],self.gps["lat"],color="pink",s=5)
        plt.xticks(rotation=-30,ha="left")
        ax = gps_ax.twinx()
        # sleep events
        for s, e in zip(self.sleep["start_time"].unique(),self.sleep["end_time"].unique()):
            ax.axvspan(pd.to_datetime(s),pd.to_datetime(e),color="grey",alpha=0.25,zorder=1)
        # beacon data
        ax.scatter(self.complete["timestamp"],self.complete["co2"],color="grey",alpha=0.5,s=10,zorder=2) # raw
        ax.scatter(self.complete["timestamp"],self.complete["sma_co2"],s=30,color="black",zorder=3) # smoothed
        ax.scatter(self.inc["timestamp"],self.inc["sma_co2"],s=25,color="seagreen",zorder=4) # increasing and smoothed
        ax.scatter(self.filtered["timestamp"],self.filtered["co2"],s=20,color="firebrick",zorder=5) # gps filtered
        ax.scatter(self.occupied["timestamp"],self.occupied["co2"],s=15,color="goldenrod",zorder=6) # co2 filtered
        ax.scatter(self.fully_filtered["timestamp"],self.fully_filtered["co2"],s=5,color="white",zorder=7) # gps or co2 filtered
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=-30,ha="left")
        
        if "time_period" in kwargs.keys():
            ax.set_xlim([kwargs["time_period"][0],kwargs["time_period"][1]])

        plt.show()
        plt.close()
            
    def run(self):
        """runs the analysis"""
        for dataset, label in zip([self.complete,self.filtered,self.prior,self.morning],["Complete","Filtered","Before Sleep","After Waking"]):
            print(label)
            self.plot_timeseries(dataset,"co2")