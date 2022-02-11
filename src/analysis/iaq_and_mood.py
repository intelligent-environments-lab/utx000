# IAQ and Mood
# -----------
# Author: Hagen
# Date: 02/209/22
# Description: 

import sys
sys.path.append('../')

# user-created libraries
from src.visualization import visualize

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

class Analyze():

    def __init__(self,data_dir="../data", study_suffix="ux_s20") -> None:
        """
        Initializing method

        Parameters
        ----------
        data_dir : str, default "../../data"
            location of the data directory in the project
        study_suffix : str, default "ux_s20"
            suffix corresponding to the study

        Creates
        -------
        ema : DataFrame
            numeric responses to the EMAs completed at home
        iaq : DataFrame
            all available beacon IAQ data
        """
        self.data_dir = data_dir
        self.suffix = study_suffix

        # EMA data
        ema = pd.read_csv(f"{self.data_dir}/processed/beiwe-ema_at_home_v2-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        for column in ema.columns:
            if column != "beiwe":
                ema[column] = pd.to_numeric(ema[column]) # sometimes the data are not numeric

        self.ema = ema
        # pre-processing ema
        self.add_discontent()
        self.binarize_ema()

        # IAQ data
        iaq = pd.read_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv',
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        iaq.drop(["beacon","redcap","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm10_mass","no2","lux","co"],axis=1,inplace=True)
        for column in iaq.columns:
            if column != "beiwe":
                iaq[column] = pd.to_numeric(iaq[column]) # might as well

        self.iaq = iaq

    def add_discontent(self):
        """
        Reverses the content score so that all moods "point" in the same direction
        """
        self.ema["discontent"] = 3 - self.ema["content"]
    
    def binarize_ema(self,moods=["content","discontent","stress","lonely","sad","energy"]):
        """
        Binarizes the ema scores

        Parameters
        ----------
        moods : list of str, default ["content","discontent","stress","lonely","sad","energy"]
            moods in ema to consider
        """
        for mood in moods:
            if mood in ["content","discontent","energy"]:
                self.ema[f"{mood}_binary"] = [0 if score < 2 else 1 for score in self.ema[mood]]
            else:
                self.ema[f"{mood}_binary"] = [0 if score == 0 else 1 for score in self.ema[mood]]

    def binarize_iaq(self,iaq_params=["co2","tvoc","pm2p5_mass","temperature_c","rh"]):
        """
        Gets the binary encoding of the vars in pollutants for each participant
        
        Parameters
        ----------
        df: dataframe with mean pollutant concentrations
        raw_ieq: dataframe of the unaltered ieq data from the entire deployment
        pollutants: list of strings corresponding to the IEQ parameters of interest - must have corresponding columns in the other two dataframes
        
        Need to rethink this method for two reasons:
        - how are we binarizing and over what window?
        - what datasets do we need and should be included?
        """ 
        df_bi = pd.DataFrame()
        for pt in self.iaq["beiwe"].unique():
            df_pt = self.iaq[self.iaq["beiwe"] == pt]
            ieq_pt = raw_ieq[raw_ieq["beiwe"] == pt]
            for param in iaq_params:
                try:
                    mean_night = np.nanmean(ieq_pt[f"{param}"])
                except KeyError as e:
                    print(f"Exiting: {e} not in the raw IEQ data")
                    return
                
                df_pt[f"{param}_binary"] = df_pt.apply(lambda x: encode_ieq(x[f"{pollutant}_mean"],mean_night), axis="columns")

            df_bi = df_bi.append(df_pt)

        return df_bi
    
    def summarize_iaq_before_submission(self, window=10, min_time_at_home=60, percentile=0.9):
        """
        Summarizes IAQ measurements for a given period prior to submission of the EMA

        This method seems a bit inefficient and poorly written but we should operate under the
        "what ain't broke, don't fix" clause
        
        Parameters
        ----------
        window : int, boolean, default 10
            Number of minutes to consider before EMA submission. If True, then uses the entire 
            time prior to submission as determined by GPS traces from Beiwe.
        min_time_at_home : int, default 60
            minimum number of seconds the participant has to be at home
        percentile : float, default 0.9
            percentile to include in summary statistics

        Creates
        -------
        ema_and_iaq : DataFrame
            original dataframe with summarized IAQ parameters
        iaq_prior : DataFrame
            iaq data from periods at home
        """
        iaq_prior = pd.DataFrame()
        # ensuring timestamp is index of ema
        try:
            self.ema.set_index("timestamp",inplace=True)
        except KeyError:
            # assuming "timestamp" is the index
            pass
        ema_and_iaq = pd.DataFrame()
        for pt in self.ema["beiwe"].unique():
            # participant-specific data
            iaq_pt = self.iaq[self.iaq["beiwe"] == pt]
            ema_pt = self.ema[self.ema["beiwe"] == pt]
            # if unecessary columns are left over
            for bad_col in ["beiwe","beacon","redcap","fitbit"]:
                try:
                    iaq_pt.drop(bad_col,axis=1,inplace=True)
                except KeyError:
                    pass
            # summary stat DFs
            iaq_prior_mean = pd.DataFrame()
            iaq_prior_median = pd.DataFrame()
            iaq_prior_max = pd.DataFrame()
            iaq_prior_sum = pd.DataFrame()
            iaq_prior_range = pd.DataFrame()
            iaq_prior_delta = pd.DataFrame()
            iaq_prior_percentile = pd.DataFrame()
            for t in ema_pt.index:
                if window == True:
                    try:
                        t_at_home = ema_pt.loc[t,"time_at_home"]
                        if t_at_home > min_time_at_home:
                            s = t - timedelta(seconds=t_at_home)
                        else:
                            s = t
                    except KeyError as e:
                        print(e)
                        return
                else:
                    s = t - timedelta(minutes=window)
                iaq_prior_pt = iaq_pt[s:t]
                if len(iaq_prior_pt) > 0:
                    iaq_prior_mean = pd.concat([iaq_prior_mean,iaq_prior_pt.mean(axis=0)],axis=1)
                    iaq_prior_median = pd.concat([iaq_prior_median,iaq_prior_pt.median(axis=0)],axis=1)
                    iaq_prior_max = pd.concat([iaq_prior_max,iaq_prior_pt.max(axis=0)],axis=1)
                    iaq_prior_range = pd.concat([iaq_prior_range,iaq_prior_pt.max(axis=0) - iaq_prior_pt.min(axis=0)],axis=1)
                    iaq_prior_delta = pd.concat([iaq_prior_delta,iaq_prior_pt.iloc[-1,:] - iaq_prior_pt.iloc[0,:]],axis=1)
                    iaq_prior_sum = pd.concat([iaq_prior_sum,iaq_prior_pt.sum(axis=0,numeric_only=True)],axis=1)
                    iaq_prior_percentile = pd.concat([iaq_prior_percentile,iaq_prior_pt.quantile(percentile,axis=0,numeric_only=True)],axis=1)
                    iaq_prior = iaq_prior.append(iaq_prior_pt)

            # mean
            iaq_prior_mean = iaq_prior_mean.T
            iaq_prior_mean.columns  = [col+"_mean" for col in iaq_prior_mean.columns]
            # median
            iaq_prior_median = iaq_prior_median.T
            iaq_prior_median.columns  = [col+"_median" for col in iaq_prior_median.columns]
            # max
            iaq_prior_max = iaq_prior_max.T
            iaq_prior_max.columns  = [col+"_max" for col in iaq_prior_max.columns]
            # sum
            iaq_prior_sum = iaq_prior_sum.T
            iaq_prior_sum.columns  = [col+"_sum" for col in iaq_prior_sum.columns]
            # range
            iaq_prior_range = iaq_prior_range.T
            iaq_prior_range.columns  = [col+"_range" for col in iaq_prior_range.columns]
            # delta
            iaq_prior_delta = iaq_prior_delta.T
            iaq_prior_delta.columns  = [col+"_delta" for col in iaq_prior_delta.columns]
            # percentile
            iaq_prior_percentile = iaq_prior_percentile.T
            iaq_prior_percentile.columns = [col+f"_{percentile}" for col in iaq_prior_percentile.columns]

            ema_iaq_pt = pd.concat([ema_pt.reset_index(),
                                    iaq_prior_mean.reset_index(drop=True),
                                    iaq_prior_median.reset_index(drop=True),
                                    iaq_prior_max.reset_index(drop=True),
                                    iaq_prior_sum.reset_index(drop=True),
                                    iaq_prior_range.reset_index(drop=True),
                                    iaq_prior_delta.reset_index(drop=True),
                                    iaq_prior_percentile.reset_index(drop=True)],axis=1)
            ema_and_iaq = ema_and_iaq.append(ema_iaq_pt)

        self.ema_and_iaq = ema_and_iaq
        self.iaq_prior = iaq_prior.dropna(subset=["co2","pm2p5_mass","temperature_c","rh"],how="all")
