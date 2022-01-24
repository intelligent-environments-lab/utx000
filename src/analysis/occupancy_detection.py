import pandas as pd
import numpy as np

import warnings

import sys
sys.path.append('../')
from src.visualization import visualize

from datetime import datetime, timedelta
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Sklearn
## classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
## evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
## other methods
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

class PreProcess:

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

    def iaq_comparison(self, iaq_param="co2", participants=None, by_id="beiwe", occ_label="occupied", unocc_label="unoccupied"):
        """
        Compares distributions of IAQ measurements between occupied and unoccupied conditions

        Parameters
        ----------
        iaq_param : str, default "co2"
            the parameters/column to use
        participants : str or list of str, default None
            participants to inspect
            None corresponds to all participants in objects data attribute
        by_id : str, default "beiwe"
            ID to prse out data by
        occ_label : str, default "occupied"
            label for occupied
        unocc_label : str, default "unoccupied"
            label for unoccupied

        Returns
        -------
        <void>
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
            occupied_iaq = data_pt[data_pt["label"] == occ_label]
            unoccupied_iaq = data_pt[data_pt["label"] == unocc_label]
            sns.kdeplot(x=iaq_param,data=occupied_iaq,
                lw=2,color="seagreen",cut=0,
                label=occ_label.title(),ax=ax)
            sns.kdeplot(x=iaq_param,data=unoccupied_iaq,
                lw=2,color="firebrick",cut=0,
                label=unocc_label.title(),ax=ax)
            # x-axis
            ax.set_xlabel(f"{visualize.get_label(iaq_param)} Concentration ({visualize.get_units(iaq_param)})",fontsize=16)
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

class Classify:

    def __init__(self, data, zero_label="unoccupied", one_label="occupied",) -> None:
        """
        Parameters
        ----------
        data : DataFrame
            pre-processed data from the PreProcess class
        zero_label : str, default "unoccupied"
            string corresponding to a label of 0
        one_label : str, default "occupied"
            string corresponding to a label of 1
        """
        self.data = data
        self.data.replace({zero_label:0, one_label:1},inplace=True)

    def create_pipeline(self, model, model_params=None):
        """
        Creates model pipeline

        Parameters
        ----------
        model : sklearn classifier, default RandomForestClassifier
            model to use for classification
        model_params : dict, default None
            parameters to use for the ML model

        Creates
        -------
        pipe : sklearn pipeline object
        """
        preprocessing_pipe = Pipeline(steps=[
            ("scale",StandardScaler())
            ])

        if model_params:
            rf = model(**model_params)
        else:
            rf = model() # default classifier

        pipe = Pipeline(steps=[
            ("preprocess", preprocessing_pipe),
            ("model", rf)
        ])

        self.pipe = pipe

    def update_params(self,model,model_params,from_run=False):
        """
        Updates the model parameters within the class pipe object
        
        Parameters
        ----------
        model : 

        model_params : dict
            classifier model parameters
        from_run : boolean, default False
            whether parameters are coming from a call to run() - have to remove prefix

        Creates
        -------
        pipe : SKlearn Pipeline
            new Pipeline object with updated model parameters
        """
        if from_run:
            model_params_unannotated = {f"{k.split('__')[1]}": v for k, v in model_params.items()}
            self.create_pipeline(model=model,model_params=model_params_unannotated)
        else:
            self.create_pipeline(model=model,model_params=model_params)

    def split(self, features=["co2"], target="label", test_size=0.33):
        """
        Creates the training and testing sets
        
        Parameters
        ----------
        features : list of str, default ["co2"]
            columns in data to use as model features
        target : str, default "label"
            column in data to use as target
        test_size : float, default 0.33
            size of the testing datasets

        Creates
        -------
        X_train : np.array
            feature training data
        X_test : np.array
            feature testing data
        y_train : np.array
            target training data
        y_test : np.array
            target training data
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[features], self.data[target],test_size=test_size, random_state=42)

    def perform_gridsearch(self, param_dict, cv=3, verbose_level=0):
        """
        Performs gridsearch on the class pipeline

        Parameters
        ----------
        param_dict : dict
            comprehensive dictionary to run the gridsearch on
        cv : int, default 3
            number of cross-validations to run in GridSearch
        verbose_level : int, default 0
            verbosity level for the GridSearchCV - see sklearn documentation for more info

        Returns
        -------
        <best_params> : dict
            best model parameters from the GridSearch
        """
        
        # adding prefix for pipeline 
        annotated_param_dict = {f"model__{k}": v for k, v in param_dict.items()}

        try:
            opt = GridSearchCV(self.pipe, annotated_param_dict, cv=cv, scoring='accuracy',verbose=verbose_level)
            opt.fit(self.X_train, self.y_train)

            print("\t\tBest Score\n\t\t\t", round(opt.best_score_,3))
            print("\t\tBest Params\n\t\t\t", opt.best_params_)

            return opt.best_params_

        except AttributeError:
            warnings.warn("Missing attribute(s) - make sure the pipeline object and training/test sets have been created")

            return None

    def make_predictions(self):
        """
        Makes the predictions on the training set with the class pipeline

        Creates
        -------
        predictions : np.array
            predicted target labels
        """
        try:
            self.pipe.fit(self.X_train,self.y_train)
            self.predictions = self.pipe.predict(self.X_test)
        except AttributeError:
            warnings.warn("Missing attribute(s) - make sure the pipeline object and training/test sets have been created")

    def make_evaluations(self):
        """
        Evaluates the performance of the model

        Returns
        -------
        res : dict
            evaluation metrics
        cm : np.array
            confusion matrix
        """
        res = {}
        try:
            cm = confusion_matrix(self.y_test, self.predictions)
            # getting evaluation metrics
            acc = accuracy_score(self.y_test, self.predictions)
            recall = recall_score(self.y_test, self.predictions)
            precision = precision_score(self.y_test, self.predictions)
            auc = roc_auc_score(self.y_test, self.predictions)
            f1 = f1_score(self.y_test, self.predictions)
            # adding metrics to res dict
            for key, val in zip(["accuracy","recall","precision","f1","roc_auc"],[acc, recall, precision, f1, auc]):
                res[key] = val

        except AttributeError:
            warnings.warn("Missing attribute(s) - make sure the training/test sets and predictions have been created")
            cm = None # so no error from return

        return res, cm

    def optimize(self, model, param_grid, features=["co2"], target="label", test_size=0.33, cv=3, verbose_level=0):
        """
        Runs classification to optimize the model parameters

        Parameters
        ----------
        model : SKlearn classifier
            which model to use for classification
        param_grid : dict
            classifier parameters to GridSearch through
        features : list of str, default ["co2"]
            columns in data to use as model features
        target : str, default "label"
            column in data to use as target
        test_size : float, default 0.33
            size of the testing datasets
        cv : int, default 3
            number of cross-validations to run in GridSearch
        verbose_level : int, default 0
            verbosity level for the GridSearchCV - see sklearn documentation for more info

        Returns
        -------
        results : dict
            evaluation results from the classification
        """
        # Result Dicts
        classification_results = {"beiwe":[],"n_occupied":[],"n_unoccupied":[],"accuracy":[],"recall":[],"precision":[],"f1":[],"roc_auc":[]}
        model_results = {k: [] for k in param_grid.keys()}
        model_results["beiwe"] = []
        model_results["n_occupied"] = []
        model_results["n_unoccupied"] = []

        # creating copy of all participants data
        data_all = self.data.copy()
        # looping through all participants to get per-participant models
        for pt in data_all["beiwe"].unique():
            # Classifying per Participant
            self.data = data_all[data_all["beiwe"] == pt]

            print("Starting...\n")
            s = datetime.now()

            print("\tCreating Pipeline")
            self.create_pipeline(model=model)
            print("\tSplitting Data")
            self.split(features=features, target=target, test_size=test_size)
            print("\tPerforming Gridsearch")
            self.best_params = self.perform_gridsearch(param_dict=param_grid, cv=cv, verbose_level=verbose_level)
            print("\tUpdating Parameters")
            self.update_params(model, self.best_params, from_run=True)
            print("\tMaking Predictions")
            self.make_predictions()
            print("\tEvaluating Classifier")
            res_pt, _ = self.make_evaluations()

            e = datetime.now()
            print(f"\nDone - Time for Evaluation: {round((e-s).total_seconds(),2)} seconds")
            
            # Classification Results
            ## adding meta data
            res_pt["beiwe"] = pt
            n_occupied = len(self.data[self.data["label"] == 1])
            n_unoccupied = len(self.data[self.data["label"] == 0])
            res_pt["n_occupied"] = n_occupied
            res_pt["n_unoccupied"] = n_unoccupied
            ## adding evaluation metrics 
            for k in res_pt.keys():
                classification_results[k].append(res_pt[k])
            
            # Model Hyperparameter Results
            ## adding meta data
            best_params_short = {f"{k.split('__')[1]}": v for k, v in self.best_params.items()}
            best_params_short["beiwe"] = pt
            best_params_short["n_occupied"] = n_occupied
            best_params_short["n_unoccupied"] = n_unoccupied
            
            # adding to model results from gridsearch
            for k in best_params_short.keys():
                model_results[k].append(best_params_short[k])

        # reset class data
        self.data = data_all
        return pd.DataFrame(classification_results), pd.DataFrame(model_results)

    def run(self, model, model_params, participants=None, features=["co2"], target="label", test_size=0.33):
        """
        Runs classification on participant-level data

        Parameters
        ----------
        model : SKlearn classifier
            which model to use for classification
        model_params : dict
            optimal classifier parameters
        features : list of str, default ["co2"]
            columns in data to use as model features
        target : str, default "label"
            column in data to use as target
        zero_label : str, default "unoccupied"
            string corresponding to a label of 0
        one_label : str, default "occupied"
            string corresponding to a label of 1
        test_size : float, default 0.33
            size of the testing datasets

        Returns
        -------
        classification_results : dict
            evaluation results from the classification
        """

        classification_results = {"beiwe":[],"n_occupied":[],"n_unoccupied":[],"accuracy":[],"recall":[],"precision":[],"f1":[],"roc_auc":[]}

        # getting list of participants
        if participants == None:
            pt_list = self.data["beiwe"].unique()
        elif isinstance(participants,list):
            pt_list = participants
        else:
            pt_list = [participants] 

        data_all = self.data.copy() # saving all the data since the methods use the class object
        for pt in pt_list:
            # Classifying per Participant
            self.data = data_all[data_all["beiwe"] == pt] # overwriting class data
            
            print("Starting...\n")
            s = datetime.now()

            print("\tCreating Pipeline")
            self.create_pipeline(model=model,model_params=model_params)
            print("\tSplitting Data")
            self.split(features=features, target=target, test_size=test_size)
            print("\tMaking Predictions")
            self.make_predictions()
            print("\tEvaluating Classifier")
            res_pt, self.cm = self.make_evaluations()

            e = datetime.now()
            print(f"\nDone - Time for Evaluation: {round((e-s).total_seconds(),2)} seconds")
            
            # Classification Results
            ## adding meta data
            res_pt["beiwe"] = pt
            n_occupied = len(self.data[self.data["label"] == 1])
            n_unoccupied = len(self.data[self.data["label"] == 0])
            res_pt["n_occupied"] = n_occupied
            res_pt["n_unoccupied"] = n_unoccupied
            ## adding evaluation metrics 
            for k in res_pt.keys():
                classification_results[k].append(res_pt[k])

        self.data = data_all # resetting the class data object 
        self.results = pd.DataFrame(classification_results)

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