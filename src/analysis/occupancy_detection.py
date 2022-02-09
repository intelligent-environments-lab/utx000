from __future__ import annotations
from re import A
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

    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data", features=["co2"], resample_rate=10, resample_nightly=True) -> None:
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
        resample_rate : int, default 15
            number of minutes to resample to
        """
        self.study = study
        self.suffix = study_suffix
        self.data_dir = data_dir

        self.features = features
        self.resample_rate = resample_rate

        # Loading Data
        # ------------
        ## GPS
        self.gps = pd.read_csv(f"{self.data_dir}/processed/beiwe-gps_beacon_pts-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        self.homestay = pd.read_csv(f"{self.data_dir}/processed/beiwe-homestay-{self.suffix}.csv",
            parse_dates=["start","end"],infer_datetime_format=True)

        ## Beacon
        beacon_all = pd.read_csv(f"{data_dir}/processed/beacon-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        beacon_all.drop(["redcap","fitbit"],axis=1,inplace=True)
        self.beacon_all = self.resample_data(beacon_all,rate=self.resample_rate)
        self.beacon_all.dropna(subset=self.features,inplace=True)
        ### Beacon during only Fitbit-detected sleep events
        self.beacon_nightly = pd.read_csv(f"{self.data_dir}/processed/beacon_by_night-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)
        self.beacon_nightly.drop(["no2","increasing_co2","ema","redcap","fitbit"],axis=1,inplace=True)
        if resample_nightly:
            self.beacon_nightly = self.resample_data(self.beacon_nightly,rate=self.resample_rate)

        self.beacon_nightly.dropna(subset=self.features,inplace=True)
        ### Beacon data from unoccupied periods (has been corss-referenced with available GPS data)
        self.set_beacon_occupancy_data()

    def set_unoccupied_bedroom_data(self, verbose=False):
        """
        Finds the IAQ data for periods when GPS data are available and participants are not at home

        CANNOT RESAMPLE BEACON NIGHTLY IF USING THIS METHOD
        """
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
            unoccupied_resampled = unoccupied_only.set_index("timestamp").resample(f"{self.resample_rate}T").mean().dropna()
            unoccupied_resampled["beiwe"] = pt # adding back in the participant ID
            if verbose:
                print(f"{pt}: {len(unoccupied_resampled)}")
            
            # merging gps data from unoccupied periods with beacon data
            iaq_unoccupied = beacon_all_pt.merge(right=unoccupied_resampled.reset_index(),on=["beiwe","timestamp"],how="inner")
            beacon_unoccupied = beacon_unoccupied.append(iaq_unoccupied)

        self.beacon_bedroom_unoccupied = beacon_unoccupied
        
    def set_beacon_occupancy_data(self,verbose=False):
        """
        
        """
        unoccupied_gps = pd.DataFrame()
        occupied_gps = pd.DataFrame()
        # looping through the participants to get GPS data when they are not home
        for pt in self.homestay["beiwe"].unique():
            # pt-specific data
            gps_pt = self.gps[self.gps["beiwe"] == pt]
            homestay_pt = self.homestay[self.homestay["beiwe"] == pt]
            # get gps data for homestay periods
            home_data_list = [gps_pt[s:e] for s, e in homestay_pt[['start','end']].to_numpy()]
            occupied_pt = pd.concat(home_data_list)
            # combine gps and homestay gps data
            temp = gps_pt.merge(occupied_pt,left_index=True,right_index=True,how="left",suffixes=("","_home"),indicator=True)
            # take only gps data for periods outside of homestay
            unoccupied_pt = temp[temp["_merge"] == "left_only"]
            # drop unnecessary columns from merge
            unoccupied_pt = unoccupied_pt[[col for col in unoccupied_pt.columns if not col.endswith("home") and not col.endswith("merge")]]
            # resample gps data
            unoccupied_pt_resampled = unoccupied_pt.resample(f"{self.resample_rate}T").mean().dropna()
            unoccupied_pt_resampled["beiwe"] = pt
            occupied_pt_resampled = occupied_pt.resample(f"{self.resample_rate}T").mean().dropna()
            occupied_pt_resampled["beiwe"] = pt
            
            # combine to aggregate df
            unoccupied_gps = unoccupied_gps.append(unoccupied_pt_resampled)
            occupied_gps = occupied_gps.append(occupied_pt_resampled)

        # merge gps data from periods away from home with beacon data
        beacon_all = self.beacon_all.copy() # making a copy so we don't ruin the original
        self.beacon_unoccupied = beacon_all.reset_index().merge(unoccupied_gps.reset_index(),on=["beiwe","timestamp"]).set_index("timestamp")
        self.beacon_occupied = beacon_all.reset_index().merge(occupied_gps.reset_index(),on=["beiwe","timestamp"]).set_index("timestamp")

    def add_bedroom_label(self,home_labels=[1],drop_duplicates=True,verbose=False):
        """
        Includes a column that corresponds to bedroom occupancy label 

        Parameters
        ----------
        home_labels : list of int, default [1]
            labels to use that indicate if the participant is home in [0,1]
            0 indicates participants were confirmed home by sleep episode and CO2/T measurements
            1 indicates participants were confirmed home by GPS/address cross-reference
        verbose : boolean, default False
            increased output for debugging purposes
        
        Attributes Created
        ------------------
        bedroom_data : DataFrame
            data that includes an occupied and unoccupied label
        """
        # occupied label
        occupied_data = self.beacon_nightly.copy() # nightly measurements are from occupied periods
        occupied_data = occupied_data[occupied_data["home"].isin(home_labels)] # ensures we use gps-confirmed data
        occupied_data["bedroom"] = "occupied" # add label

        # adding labels, combining, and dropping any pesky duplicates
        occupied_data["bedroom"] = "occupied"
        unoccupied_data = self.beacon_unoccupied.copy()
        unoccupied_data["bedroom"] = "unoccupied"
        labeled_data = occupied_data.append(unoccupied_data)
        if drop_duplicates:
            labeled_data.drop_duplicates(subset=["beiwe","co2"],inplace=True)
        labeled_data.reset_index(inplace=True)
        
        # ensuring participants have both occupied and unoccupied data
        pts_with_one_label = []
        for pt in labeled_data["beiwe"].unique():
            data_pt = labeled_data[labeled_data["beiwe"] == pt]
            if len(data_pt["bedroom"].unique()) != 2:
                pts_with_one_label.append(pt)
        labeled_data = labeled_data[~labeled_data["beiwe"].isin(pts_with_one_label)]

        self.bedroom_data = labeled_data.set_index("timestamp")

    def add_home_label(self,occupied_label="occupied",unoccupied_label="unoccupied",drop_duplicates=True,verbose=False):
        """
        Includes a column that corresponds to home occupancy 
        
        Parameters
        ----------
        occupied_label : str or int, default "occupied"
            label to use for occupied periods
        unoccupied_label : str or int, default "unoccupied"
            label to use for unoccupied periods

        Attributes Created
        ------------------
        home_data : DataFrame
            IAQ data with labels indicating if participants are home or not
        """
        # occupied data
        occupied_data = self.beacon_occupied.copy()
        occupied_data["home"] = occupied_label
        # unoccupied data
        unoccupied_data = self.beacon_unoccupied.copy()
        unoccupied_data["home"] = unoccupied_label
        # combining unoccupied and occupied data, dropping duplicates and saving as new attribute
        home_data = occupied_data.append(unoccupied_data)
        if drop_duplicates:
            home_data.drop_duplicates(subset=["beiwe","co2"],inplace=True)
        home_data.reset_index(inplace=True)

        # ensuring participants have both occupied and unoccupied data
        pts_with_one_label = []
        for pt in home_data["beiwe"].unique():
            data_pt = home_data[home_data["beiwe"] == pt]
            if len(data_pt["home"].unique()) != 2:
                pts_with_one_label.append(pt)
        home_data = home_data[~home_data["beiwe"].isin(pts_with_one_label)]

        self.home_data = home_data.set_index("timestamp")

    def set_data(self,bedroom=True):
        """
        Sets the class data attribute

        Parameters
        ----------
        bedroom : boolean, default True
            whether to use bedroom data or home data if False

        Creates Attribute
        -----------------
        data : DataFrame
            main data to use for further analysis
        """
        if bedroom:
            try:
                self.data = self.bedroom_data
            except AttributeError:
                print("Need to create bedroom labeled data")
        else:
            try:
                self.data = self.home_data
            except AttributeError:
                print("Need to create home labeled data")

    def resample_data(self, df, rate=10, by_id="beiwe"):
        """
        Resamples data to the given rate

        Parameters
        ----------
        df : DataFrame
            data to be resampled - index must be datetime
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
        for pt in df[by_id].unique():
            data_pt = df[df[by_id] == pt]
            resampled_pt = data_pt.resample(f"{rate}T").mean()
            resampled_pt[by_id] = pt # adding ID back in
            resampled = resampled.append(resampled_pt)

        return resampled

    def iaq_comparison(self, iaq_param="co2", participants=None, by_id="beiwe", space_label="bedroom", occ_label="occupied", unocc_label="unoccupied"):
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
        space_label : str, default "bedroom"
            specifies column to use for occupancy
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
            occupied_iaq = data_pt[data_pt[space_label] == occ_label]
            unoccupied_iaq = data_pt[data_pt[space_label] == unocc_label]
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

    def __init__(self, data, features=["co2"], zero_label="unoccupied", one_label="occupied",) -> None:
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
        self.features = features
        self.data = data.dropna(subset=features)
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

    def split(self, target="bedroom", test_size=0.33):
        """
        Creates the training and testing sets
        
        Parameters
        ----------
        features : list of str, default ["co2"]
            columns in data to use as model features
        target : str, default "bedroom"
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.features], self.data[target],test_size=test_size, random_state=42)

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

    def optimize(self, model, param_grid, target="label", test_size=0.33, cv=3, verbose_level=0):
        """
        Runs classification to optimize the model parameters on a participant-level

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

        # removing 

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
            self.split(target=target, test_size=test_size)
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
            n_occupied = len(self.data[self.data[target] == 1])
            n_unoccupied = len(self.data[self.data[target] == 0])
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

    def run(self, model, model_params, participants=None, target="label", test_size=0.33):
        """
        Runs classification on participant-level data

        Parameters
        ----------
        model : SKlearn classifier
            which model to use for classification
        model_params : dict
            optimal classifier parameters
        participants : list of str, default None
            participants to consider - if None, then use all participants in the class data
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

        classification_results = {"beiwe":[],"n_occupied":[],"n_unoccupied":[],"runtime":[],"accuracy":[],"recall":[],"precision":[],"f1":[],"roc_auc":[]}

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
            self.split(target=target, test_size=test_size)
            print("\tMaking Predictions")
            self.make_predictions()
            print("\tEvaluating Classifier")
            res_pt, self.cm = self.make_evaluations()

            e = datetime.now()
            runtime = round((e-s).total_seconds(),2)
            print(f"\nDone - Time for Evaluation: {runtime} seconds")
            
            # Classification Results
            ## adding meta data
            res_pt["beiwe"] = pt
            n_occupied = len(self.data[self.data[target] == 1])
            n_unoccupied = len(self.data[self.data[target] == 0])
            res_pt["n_occupied"] = n_occupied
            res_pt["n_unoccupied"] = n_unoccupied
            res_pt["runtime"] = runtime
            ## adding evaluation metrics 
            for k in res_pt.keys():
                classification_results[k].append(res_pt[k])

        self.data = data_all # resetting the class data object 
        self.results = pd.DataFrame(classification_results)

    def classify(self, model, model_params, observations, target="bedroom", verbose=0):
        """
        Classifies occupancy on unlabeled data

        Parameters
        ----------
        model : SKlearn classifier
            which model to use for classification
        model_params : dict
            optimal classifier parameters
        observations : DataFrame
            unlabeled data to classify on
        features : list of str, default ["co2"]
            columns in data to use as model features
        target : str, default "bedroom"
            column in data to use as target
        verbose : int, default 0


        Returns
        -------

        """

        data_all = self.data.copy() # saving all the data since the methods use the class object
        probs = pd.DataFrame()
        for pt in observations["beiwe"].unique():
            # Classifying per Participant
            self.data = data_all[data_all["beiwe"] == pt] # overwriting class data
            observations_pt = observations[observations["beiwe"] == pt]
            if len(self.data) > 0:
                print("Starting...\n")
                s = datetime.now()

                print("\tCreating Pipeline")
                self.create_pipeline(model=model,model_params=model_params)
                print("\tFitting Model to Labeled Data")
                self.pipe.fit(self.data[self.features],self.data[target])
                print("\tClassifying Occupancy on Unlabeled Data")
                classifications = self.pipe.predict_proba(observations_pt[self.features])

                e = datetime.now()
                print(f"\nDone - Time for Evaluation: {round((e-s).total_seconds(),2)} seconds")

                probs = probs.append(pd.DataFrame(classifications))
            else:
                # update observations by removing the participant
                if verbose > 0:
                    print("No available labeled data for participant {pt}")
                observations = observations[observations["beiwe"] != pt]

        self.data = data_all # resetting the class data object 
        self.classifications = pd.concat([observations.reset_index(),probs.reset_index()],axis=1).set_index("timestamp")[["beiwe"]+self.features+[0,1]]

    def get_classification_color(self,percent,highest=0.9):
        """
        Returns color based on the percent level
        """
        if percent >= highest:
            return "seagreen"
        elif percent >= 0.8:
            return "goldenrod"
        elif percent >= 0.7:
            return "orange"
        elif percent >= 0.6:
            return "firebrick"
        else: # default color is black
            return "black"

    def ts_prediction(self,participants=None,percents=[0.6,0.7,0.8,0.9],iaq_param="co2",save=False,**kwargs):
        """
        Plots timeseries of IAQ data, colored by occupancy

        Parameters
        ----------
        participants : list of str, default None
            participants to consider - if None, then use all participants in the class data

        """
        # checking to see if classifications have been made
        try:
            _ = self.classifications.copy()
        except AttributeError:
            print("Need to create classification predictions using the `classify` method")
            return

        # getting list of participants
        if participants == None:
            pt_list = self.classifications["beiwe"].unique()
        elif isinstance(participants,list):
            pt_list = participants
        else:
            pt_list = [participants]

        _, axes = plt.subplots(len(pt_list),1,figsize=(16,4*len(pt_list)))
        try:
            _ = len(axes)
        except TypeError:
            axes = [axes]
        for pt, ax in zip(pt_list,axes):
            # pt-specific data
            pt_class = self.classifications[self.classifications["beiwe"] == pt]
            # coloring points based on percentages
            for percent in percents:
                above_percent = pt_class[pt_class[1] >= percent]
                ax.scatter(above_percent.index,above_percent[iaq_param],s=5,color=self.get_classification_color(percent),label=f"> {percent}",zorder=percent*10)
            # plotting remaining data
            ax.plot(pt_class.index,pt_class[iaq_param],color="black",lw=2,label=f"< {np.min(percents)}",zorder=1)
            # x-axis
            ax.set_xlabel("")
            ax.set_xlim([pt_class.index[0],pt_class.index[-1]])

            if "start_time" in kwargs.keys():
                ax.set_xlim(left=kwargs["start_time"])
            if "end_time" in kwargs.keys():
                ax.set_xlim(right=kwargs["end_time"])
            # y-axis
            ax.set_ylabel(f"{visualize.get_label(iaq_param)} ({visualize.get_units(iaq_param)})",fontsize=16)
            if iaq_param == "co2":
                ax.set_ylim([400,2500])
            # remainder
            ax.tick_params(labelsize=14)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)

            if save:
                plt.savefig(f"../reports/figures/predicted_occupancy-{pt}.png")
        
        plt.show()
        plt.close()

    def summarize_classification(self,available_data,percents=[0.7,0.8,0.9]):
        """
        Summarizes the classification abiliity by participant and provided percent cutoffs

        Parameters
        ----------
        available_data : DataFrame
            all the possible IAQ data for each participant
        percents : list of float, default [0.7,0.8,0.9]
            decimal percentages for cutoff
    
        Returns
        -------
        <res> : DataFrame
            summarized results for cutoffs
        """
        # creatiing results dictionary
        res = {f"> {p*100}%": [] for p in percents}
        res["Participant"] = []
        # looping through pts and percentages
        for pt in self.classifications["beiwe"].unique():
            # pt data
            res["Participant"].append(pt)
            available_pt = available_data[available_data["beiwe"] == pt]
            classifications_pt = self.classifications[self.classifications["beiwe"] == pt]
            if len(classifications_pt) == 0:
                for p in percents:
                    res[p].append("0 (0%)")
            else:
                for p in percents:
                    # above data
                    above_percent = classifications_pt[classifications_pt[1] > p]
                    n = len(above_percent)
                    percent_recovered = round(n / len(available_pt) * 100,1)
                    res[f"> {p*100}%"].append(f"{n} ({percent_recovered}%)")

        return pd.DataFrame(res)
                
    def save_classifications(self,model_name,annot):
        """
        
        """
        self.classifications.to_csv(f"../data/results/{model_name}-occupancy_classification-{annot}-ux_s20.csv")

class CompareModels():

    def __init__(self, model_results) -> None:
        """
        Parameters
        ----------
        model_results : dict
            keys correspond to model and values are dataframes with participant-based evaluation metrics
        """
        self.res = model_results

    def get_model_name(self,model_name_short):
        """
        Gets the model's full name from the abbreviation
        """
        abb = model_name_short.lower()
        if abb == "lr":
            return "Logistic Regression"
        elif abb == "nb":
            return "Naive Bayes"
        elif abb == "rf":
            return "Random Forest"
        elif abb == "mlp":
            return "MLP"
        else:
            return ""

    def get_symbol(self,model_name_short):
        """
        Gets the symbol for the given model
        """
        abb = model_name_short.lower()
        if abb == "lr":
            return "o"
        elif abb == "nb":
            return "d"
        elif abb == "rf":
            return "s"
        elif abb == "mlp":
            return "*"
        else:
            return ""

    def scatter_metric(self,metric="accuracy",anonymize=False, save=False, annot=None):
        """
        Scatters the evaluation metric for each participant from each method

        Parameters
        ----------
        metric : str, default "accuracy"
            evaluation metric to plot
        anonymize : boolean, default False
            use participant IDs or not

        Returns
        -------
        comb_sorted : DataFrame
            accuracy for each model for each participant
        """
        
        # combinig results so we can sort them
        df_list = []
        for m in self.res.keys():
            df_list.append(self.res[m][metric])

        comb = pd.concat(df_list,axis=1)
        comb.set_index(self.res[m]["beiwe"],inplace=True)
        comb["mean"] = comb.mean(axis=1)
        comb_sorted = comb.sort_values("mean")
        if anonymize:
            pids = comb_sorted.index
            comb_sorted.reset_index(drop=True,inplace=True)
            comb_sorted.index = [str(i) for i in range(1,len(comb_sorted)+1)]
        # plotting
        _, ax = plt.subplots(figsize=(12,4))
        for col, model in zip(range(len(comb_sorted.columns)-1),self.res.keys()):
            ax.scatter(comb_sorted.index,comb_sorted.iloc[:,col],
                marker=self.get_symbol(model), s=75,alpha=0.7,label=self.get_model_name(model))

        # x-axis
        ax.set_xlabel("ID",fontsize=16)
        ax.tick_params(axis="x",labelsize=14)
        # y-axis
        ax.set_ylabel(metric.title(),fontsize=16)
        ax.tick_params(axis="y",labelsize=14)
        ax.set_ylim(bottom=0.4,top=1)
        # remainder
        ax.legend(loc="upper center", bbox_to_anchor=(0.75,0.3),frameon=True,fontsize=14,ncol=2)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

        if save:
            if annot:
                plt.savefig(f"../reports/figures/occupancy_detection-{annot}-{metric}.pdf")
            else:
                plt.savefig(f"../reports/figures/occupancy_detection-{metric}.pdf")

        plt.show()
        plt.close()

        # adding back ids 
        if anonymize:
            comb_sorted["beiwe"] = pids

        return comb_sorted

    def compare_runtimes(self):
        """
        Compares runtime metrics
        """
        print("runtime:".upper())
        for model in self.res.keys():
            print("\t",model.upper())
            rts = self.res[model]['runtime']
            print(f"\t\tMean:\t{np.mean(rts)} s\n\t\tSum:\t{np.sum(rts)} s")

    def compare_f1s(self):
        """
        Compares F1 scores
        """
        print("F1:")
        for model in self.res.keys():
            print("\t",model.upper())
            f1s = self.res[model]['f1']
            print(f"\t\tMean:\t{np.mean(f1s)}\n\t\tSTD:\t{np.std(f1s)}")

    def save_results(self,annot=None):
        """
        Saves model results as separate csv files
        """
        for model in self.res.keys():
            if annot:
                self.res[model].to_csv(f"../data/results/{model}-metrics-{annot}-ux_s20.csv")
            else:
                self.res[model].to_csv(f"../data/results/{model}-metrics-ux_s20.csv")

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