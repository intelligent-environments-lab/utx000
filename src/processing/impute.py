
from numpy.lib.scimath import sqrt
import pandas as pd
import numpy as np
from pandas.core import frame
import random
import missingno as msno
from datetime import datetime
import json

# Iterative Imputer for MICE
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Plotting
import matplotlib.pyplot as plt

class TestData:

    def __init__(self,pt,data_dir="../",params=["co2","pm2p5_mass","pm10_mass","tvoc","temperature_c","co"],**kwargs):
        self.pt = pt
        self.data_dir = data_dir
        self.params = params

        self.data = self.import_example_data(self.pt,self.data_dir,self.params,**kwargs)

    def import_example_data(self,pt,data_dir="../",params=["co2","pm2p5_mass","pm10_mass","tvoc","temperature_c","co"],**kwargs):
        """
        Imports example data
        
        Parameters
        ----------
        pt : str
            id of the participant to import for
        data_dir : str
            path to the "data" directory in the utx000 project
        params : list, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            
        Returns
        -------
        data : DataFrame
            exemplary iaq data from participant
        """
        
        try:
            data = pd.read_csv(f"{data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",
                            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
            if "start_time" in kwargs.keys():
                start_time = kwargs["start_time"]
            else:
                start_time = data.index[0]
            if "end_time" in kwargs.keys():
                end_time = kwargs["end_time"]
            else:
                end_time = data.index[-1]
            data = data[start_time:end_time]
            idx = pd.date_range(start_time,end_time,freq="2T")
            data = data.reindex(idx,fill_value=np.nan)
            data.index.name = "timestamp"
        except FileNotFoundError:
            print("No filename for participant", pt)
            return pd.DataFrame()
        
        return data[params]

    def remove_at_random(self,df,percent=10):
        """
        Removes random rows from the dataset
        
        Parameters
        ----------
        df : DataFrame
            original data
        percent : int or float, default 10
            percent of data to remove
            
        Returns
        -------
        df_subset : DataFrame
            original data with rows removed
        """
        remove_n = int(percent/100*len(df))
        drop_indices = np.random.choice(df.index, remove_n, replace=False)
        df_subset = df.drop(drop_indices)
        
        return df_subset

    def remove_at_random_all(self,df_in,percent=10,params=["co2","pm2p5_mass","pm10_mass","tvoc","temperature_c","co"]):
        """
        Removes the given percentage of data individually across all parameters
        
        Parameters
        ----------
        df_in : DataFrame
            original data
        percent : int or float, default 10
            percent of data to remove
        params : list of str, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            parameters to remove data from
            
        Returns
        -------
        df : DataFrame
            original data with observations removed
        """
        df = df_in.copy()
        true_percent = percent / len(params)
        for param in params:
            df_param = self.remove_at_random(df_in,percent=true_percent)
            df[param] = df_param[param]
            
        return df

    def get_n_consecutive_ixs(self,ixs,n=5,perc=10):
        """
        Gets n-consecutive indices at random (Kingsley)
        
        Parameters
        ----------
        ixs : list
            consecutive indices from DataFrame
        n : int, default 5
            number of consecutive rows to remove in a single instance
        percent : int or float, default 10
            determines the number of instances to remove
            
        Returns
        -------
        ixs : list
            new indices after removing consecutive values
        """
        ixs_count = len(ixs)*perc/100
        set_count = int(ixs_count/n)
        choice_ixs = [ix for ix in range(0,len(ixs)-(n+1),n)]
        choice_ixs = random.choices(choice_ixs,k=set_count)
        ixs_sets = [[ix + i for i in range(n)] for ix in choice_ixs]
        ixs = [ix for ixs in ixs_sets for ix in ixs]
        
        return ixs

    def remove_n_consecutive_with_other_missing(self,df_in,n,percent_all,percent_one,params,target):
        """
        Removes random periods of data from the dataset
        
        Parameters
        ----------
        df_in : DataFrame
            original dataset
        n : int
            number of consecutive rows to remove in a single instance
        percent : int or float
            determines the number of instances to remove
            
        Return
        ------
        <subset> : DataFrame
            original dataframe with the missing data
        """
        # remove data at random from all other columns
        df_missing = self.remove_at_random_all(df_in,percent=percent_all,params=params)
        
        # remove the consecutive observations 
        df = df_in.reset_index()
        drop_indices = self.get_n_consecutive_ixs(df.index,n=n,perc=percent_one)
        df_consecutive_missing = df.drop(drop_indices)
        
        # merge missing at random with missing observations for target and return
        comb = df_missing.drop(target,axis="columns").reset_index().merge(right=df_consecutive_missing[["timestamp",target]],on="timestamp",how="left")
        return comb

    # Evaluating
    def check_missing(self, df, save=False):
        """
        Checks the missing data from the given df
        """
        print("Percent of Missing Data:",round(df.isnull().sum().sum()/len(df)*100,3))
        print("From each column:")
        for param in df.columns:
            print(f"\t{param}:",round(df[param].isnull().sum()/len(df)*100,3))

        _, ax = plt.subplots()
        msno.matrix(df, ax=ax)

        if save:
            plt.savefig("/Users/hagenfritz/Desktop/missing_check.pdf",bbox_inches="tight")

        plt.show()
        plt.close()

    def visualize(self, df, param="co2", **kwargs):
        """
        Provides a timeseries of the given parameter in df
        """
        # checking for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Index is not datetime")
            return 

        # plotting
        _, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index,df[param],lw=2,color="black")
        # x-axis
        if "start_time" in kwargs.keys():
            ax.set_xlim(left=kwargs["start_time"])
        if "end_time" in kwargs.keys():
            ax.set_xlim(right=kwargs["end_time"])
        # remainder
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

        plt.show()
        plt.close()

    # setters
    def restrict_data(self,start_time,end_time):
        """
        Restricts the class data to a certain time interval
        """
        self.data = self.data[start_time:end_time]
        idx = pd.date_range(start_time,end_time,freq="2T")
        self.data = self.data.reindex(idx,fill_value=np.nan)
        self.data.index.name = "timestamp"

class Impute:

    def __init__(self,pt,data_dir,freq="2T",consecutive=False,prompt=False,**kwargs):
        """
        Impute class for BEVO Beacon data

        Parameters
        ----------

        """
        # Class Vars
        # ----------
        self.pt = pt
        self.data_dir = data_dir
        self.freq = freq
        # Loading Data
        # ------------
        # Missing Data
        if prompt:
            percent = input("Percent: ") # common to both
            self.param = input("Parameter: ")
            if consecutive:
                period = input("Period (in minutes): ")
                self.load_data_consecutive_random(percent,period,**kwargs)
            else:
                self.load_data_random(percent,**kwargs)
        # Base Data
        self.base = pd.read_csv(f"{self.data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",parse_dates=["timestamp"],
                                index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)

        if "start_time" in kwargs.keys():
            start_time = kwargs["start_time"]
        else:
            start_time = self.base.index[0]
        if "end_time" in kwargs.keys():
            end_time = kwargs["end_time"]
        else:
            end_time = self.base.index[-1]

        self.base = self.base[start_time:end_time]
        idx = pd.date_range(start_time,end_time,freq="2T")
        self.base = self.base.reindex(idx,fill_value=np.nan)
        self.base.index.name = "timestamp"

# methods for loading data
    def load_data_random(self,percent,**kwargs):
        """
        Loads in the randomly removed data
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_all-{self.param}-p{percent}-{self.pt}.csv",parse_dates=["timestamp"],
                                    index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)
        
            if "start_time" in kwargs.keys():
                start_time = kwargs["start_time"]
            else:
                start_time = self.missing.index[0]
            if "end_time" in kwargs.keys():
                end_time = kwargs["end_time"]
            else:
                end_time = self.missing.index[-1]

            self.missing = self.missing[start_time:end_time]
            idx = pd.date_range(start_time,end_time,freq="2T")
            self.missing = self.missing.reindex(idx,fill_value=np.nan)
            self.missing.index.name = "timestamp"
        except FileNotFoundError as e:
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-{self.param}-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tparam:\t{self.param}\n\tpercent:\t{percent}")

    def load_data_consecutive_random(self,percent,period,**kwargs):
        """
        Loads in the randomly removed, consecutive data for class param
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_periods_all-{self.param}-p{percent}-{period}mins-{self.pt}.csv",
                                    parse_dates=["timestamp"],index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)
            if "start_time" in kwargs.keys():
                start_time = kwargs["start_time"]
            else:
                start_time = self.missing.index[0]
            if "end_time" in kwargs.keys():
                end_time = kwargs["end_time"]
            else:
                end_time = self.missing.index[-1]

            self.missing = self.missing[start_time:end_time]
            idx = pd.date_range(start_time,end_time,freq="2T")
            self.missing = self.missing.reindex(idx,fill_value=np.nan)
            self.missing.index.name = "timestamp"
        except FileNotFoundError:
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tparam:\t{self.param}\n\tpercent:\t{percent}\n\tperiod:\t{period}")

# methods for imputing
    def interp(self,set_for_class=False):
        """
        Imputes missing data with a linear interpolation method

        Parameters
        ----------
        """
        self.interp_imputed = self.missing.copy()
        self.interp_imputed[self.param] = self.interp_imputed[self.param].interpolate().fillna(method="bfill").fillna(method="ffill")

        if set_for_class:
            self.set_imputed(self.interp_imputed)

    def mice(self,estimator=None,max_iter=30,set_for_class=False):
        """
        Imputes missing data with Mutiple Iterations of Chained Equations (MICE)
        
        Parameters
        ----------
        estimator : sklearn estimator, default None (Bayesion Ridge)
            estimator to use for IterativeImputer - see https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html?highlight=iterative%20imputer#sklearn.impute.IterativeImputer
        max_iter : int, default 30
            max number of iterations 
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        imp = IterativeImputer(estimator=estimator,max_iter=max_iter,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.mice_imputed = pd.DataFrame(imp.transform(self.missing),index=self.missing.index,columns=self.missing.columns)
        self.mice_imputed[self.mice_imputed < 0] = 0
        
        if set_for_class:
            self.set_imputed(self.mice_imputed)

    def miss_forest(self,n_estimators=100,max_depth=50,max_iter=30,set_for_class=False):
        """
        Imputes missing data with missForest

        Parameters
        ----------
        n_estimators : int, default 100
            number of trees in the forest
        max_depth : int, default 50
            max number of levels in the tree
        max_iter : int, default 30
            max number of iterations 
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,warm_start=True,random_state=42),
                                                                max_iter=max_iter,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.rf_imputed = pd.DataFrame(imp.transform(self.missing),index=self.missing.index,columns=self.missing.columns)
        self.rf_imputed[self.rf_imputed < 0] = 0

        if set_for_class:
            self.set_imputed(self.rf_imputed)

    def arima(self,set_for_class=False,return_metrics=False):
        """
        Imputes missing data with Auto-Regressive Integrated Moving Average 

        Parameters
        ----------
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        # getting order
        try:
            order = self.arima_order
        except AttributeError:
            print("No ARIMA order set, defaulting to (1,1,1)")
            order = (1,1,1)

        imp = ARIMA(self.missing[self.param], order=order, freq=self.freq, enforce_stationarity=False)
        self.arima_imputed = self.missing.copy()
        try:
            fitted = imp.fit()
            self.arima_imputed[self.param] = fitted.predict()
            self.arima_imputed[self.param].replace(0,np.nanmean(self.arima_imputed[self.param]),inplace=True)
            self.arima_imputed[self.arima_imputed < 0] = 0 # setting negative values to 0
            if set_for_class:
                self.set_imputed(self.arima_imputed)

            if return_metrics:
                return fitted.aic, fitted.bic, fitted.hqic
        except Exception as e:
            print("Error with ARIMA")
            print(e)

# setters
    def set_base(self,base):
        """
        Sets the class base data
        """
        self.base = base.asfreq('2T')

    def set_missing(self,missing):
        """
        Sets the class missing data
        """
        self.missing = missing.asfreq('2T')

    def set_imputed(self,imputed):
        """
        Sets the class imputed variable to the given data
        """
        self.imputed = imputed

    def set_param(self,param):
        """
        Sets the class param
        """
        self.param = param

    def set_arima_order(self,order=(1,1,1)):
        """
        Sets the class ARIMA parameters
        """
        self.arima_order = order

# methods for evaluating
    def evaluate(self,imputed,plot=False):
        """
        Evaluates the imputed data to the base data using the following metrics:
        - correlation coefficient: simple and easy to understand
        - mean absolute error: typical metric used in ML domain
        - root mean square error: another typical metrics used in the ML domain
        - index of agreement: less used metric to measure reliability vs validity

        Parameters
        ----------
        imputed : DataFrame
            imputed data - should be same length as base data

        Returns
        -------
        r2 : float
            pearson correlation coefficient
        mae : float
            mean absolute error
        rmse : float
            root mean square error
        ia : float
            index of agreement
        """
        # Debugging
        #print("Missing:",len(self.missing[self.missing[self.param].isnull()]))
        #print("Base:",len(self.base))
        #print("Percent:",len(self.missing[self.missing[self.param].isnull()])/len(self.base))
        # placeholders for true/base and imputed values
        y_true = self.base.loc[self.missing[self.missing[self.param].isnull()].index.values,:][self.param].values
        y_pred = imputed.loc[self.missing[self.missing[self.param].isnull()].index.values,:][self.param].values
        # Metrics
        # -------
        try:
            r2 = r2_score(y_true=y_true,y_pred=y_pred)
            mae = mean_absolute_error(y_true=y_true,y_pred=y_pred)
            rmse = sqrt(mean_squared_error(y_true=y_true,y_pred=y_pred))
            # looping through values for index of agreement
            num = 0
            den = 0
            for obs, pred in zip(y_true,y_pred):
                num += (pred - obs)**2
                den += (abs(pred - np.mean(y_true)) + abs(obs - np.mean(y_true)))**2
            ia = 1 - num/den
        except Exception as e:
            print("Error evaluating one of the metrics")
            print(e)
            r2 = 0
            mae = np.nan
            rmse = np.nan
            ia = 0

        if plot:
            _, ax = plt.subplots(figsize=(5,5))
            ax.scatter(y_true,y_pred,color="black",s=7,alpha=0.2)

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            ax.text(0.1,0.9,f"n={len(y_pred)}",transform=ax.transAxes)

            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)

            plt.show()
            plt.close()

        return r2, mae, rmse, ia

    def compare_ts(self,imputed):
        """
        Compares the imputed values to the base
        """
        _, ax = plt.subplots(figsize=(25,5))
        ax.plot(self.base.index,self.base[self.param],lw=10,color="black",label="Actual")
        ax.plot(self.missing.index,self.missing[self.param],lw=5,color="firebrick",label="Missing")
        ax.plot(imputed.index,imputed[self.param],lw=2,color="goldenrod",label="Predicted")
        
        # formatting
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend(frameon=False)
        
        plt.show()
        plt.close()

    def compare_methods(self,results,save=False,annot="",**kwargs):
        """
        Compares the metrics from multiple imputation methods
        
        Parameters
        ----------
        results : dictionary
            containts metric results for each method from run
        save : boolean, default False
            whether or not to save the figure
        
        Returns
        -------
        <void>
        """
        fig, axes = plt.subplots(1,4,figsize=(18,4),gridspec_kw={"wspace":0.2})
        for metric, ax in zip(["Pearson Correlation","MAE","RMSE","Index of Agreement"],axes):
            for method,color in zip(results.keys(),["black","cornflowerblue","seagreen","firebrick"]):
                method_res = results[method]
                try:
                    x = method_res["Percent"]
                except KeyError:
                    x = method_res["Period"]
                ax.plot(x,method_res[metric],
                        lw=2,color=color,label=method)
                # Formatting
                # ----------
                # x-axis
                if "xlim" in kwargs.keys():
                    ax.set_xlim(kwargs["xlim"])
                else: #default
                    ax.set_xlim([0,50])
                    ax.set_xticks(np.arange(0,55,5))
                # y-axis
                if metric in ["Pearson Correlation","Index of Agreement"]:
                    ax.set_ylim([0,1.0])
                # remainder
                ax.tick_params(labelsize=12)
                ax.set_title(metric,fontsize=16)
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
                
        # legend
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines,labels,loc="upper center",bbox_to_anchor=(0.5,0),frameon=False,ncol=5,fontsize=14)
        # common x-axis
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        if "xlabel" in kwargs.keys():
            plt.xlabel(kwargs["xlabel"],labelpad=8,fontsize=14)
        else:
            plt.xlabel("Percent Missing Data",labelpad=8,fontsize=14)
            
        if save:
            if annot != "":
                annot = "-" + annot
            plt.savefig(f"{self.data_dir}reports/figures/imputation/remove_at_random-method_metric_comparision{annot}.pdf",bbox_inches="tight")
        plt.show()
        plt.close()

# runners
    def run_at_random(self,param="co2",percents=[5,10,15,20,25,30,35,40,45,50],n_cv=3,verbose=False,**kwargs):
        """
        Evaluates and compares the imputation models

        Parameters
        ----------
        percents : range or list, default [5,10,15,20,25,30,35,40,45,50]
            percents to consider - must have an accompanying data file
        param : str, default "co2"
            specifies which parameter to run the analysis for
        n_cv : int, default 3
            number of cross-validation iterations
        verbose : boolean, default False
            whether or not to display output at each method iteration

        Returns
        -------
        res : dictionary
            metric results for each method
        """
        self.param = param
        # missing data generator class
        missing_gen = TestData(pt=self.pt,data_dir=self.data_dir)
        self.set_base(missing_gen.data) # setting the base to the restricted data
        # restricted dataset if given keywords
        if "start_time" in kwargs.keys():
            start_time = kwargs["start_time"]
        else:
            start_time = missing_gen.data.index[0]
        if "end_time" in kwargs.keys():
            end_time = kwargs["end_time"]
        else:
            end_time = missing_gen.data.index[-1]
        missing_gen.restrict_data(start_time,end_time)
        # defining the parameters 
        param_list = list(missing_gen.data.columns.values).copy()
        param_list.remove(param)
        res = {}
        for method, label in zip([self.interp, self.mice, self.miss_forest, self.arima],["Linear Interpolation","MICE","missForest","ARIMA"]):
            method_res = {"Percent":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[]}
            if verbose:
                print(label)
            for p in percents:
                if verbose:
                    print("\tPercent:",p)
                raw_res = {"r2":[], "mae":[], "rmse":[], "ia":[]} 
                for i in range(n_cv):
                    iteration_start = datetime.now()
                    if verbose:
                        print("\t\tIteration:",i,end="")
                    some_missing = missing_gen.remove_at_random_all(missing_gen.data,percent=10,params=param_list) # removing fixed number from all but one param
                    missing = missing_gen.remove_at_random_all(some_missing,percent=p,params=[self.param]) # removing the given percentage from the select param
                    self.set_missing(missing)
                    method(set_for_class=True)
                    try:
                        for metric, val in zip(raw_res.keys(),self.evaluate(self.imputed)):
                            raw_res[metric].append(val)
                    except Exception as e:
                        print(f"{param} - {p}")
                        print(e)
                    iteration_end = datetime.now()
                    if verbose:
                        print(f" - {round((iteration_end - iteration_start).total_seconds(),1)}")

                # averaging results from iterations
                avg_res = list(pd.DataFrame(raw_res).mean().values)
                avg_res.insert(0,p)
                # appending average results to overall results
                for metric, val in zip(method_res.keys(),avg_res):
                    method_res[metric].append(val)

            res[label] = method_res

        return res

    def run_periods_at_random(self, param="co2", percents=[5,10,15,20,30,35,40,45,50],period=60,n_cv=3,verbose=False):
        """
        Evaluates and compares imputation models on the consecutive missing observations datasets

        Parameters
        ----------
        param : str, default "co2"
            specifies which parameter to run the analysis for
        percents : range or list, default [5,10,15,20,25,30,35,40,45,50]
            percents to consider - must have an accompanying data file
        period : int, default 60
            length in minutes of the periods that have been removed
        n_cv : int, default 3
            number of cross-validation iterations
        verbose : boolean, default False
            whether or not to display output at each method iteration

        Returns
        -------
        res : dictionary
            metric results for each method
        """
        self.param = param
        res = {}
        for method, label in zip([self.mice, self.miss_forest, self.arima],["MICE","missForest","ARIMA"]):
            method_res = {"Percent":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[]}
            for p in percents:
                self.load_data_consecutive_random(percent=p,period=period)
                raw_res = {"r2":[], "mae":[], "rmse":[], "ia":[]} 
                for _ in range(n_cv):
                    method(set_for_class=True)
                    try:
                        for metric, val in zip(raw_res.keys(),self.evaluate(self.imputed)):
                            raw_res[metric].append(val)
                    except Exception as e:
                        print(f"{param} - {period} - {p}")
                        print(e)

                # averaging results from iterations
                avg_res = list(pd.DataFrame(raw_res).mean().values)
                avg_res.insert(0,p)
                # appending average results to overall results
                for metric, val in zip(method_res.keys(),avg_res):
                    method_res[metric].append(val)
                    
                if verbose:
                    print(f"Method: {label} - Period: {period} - Percent: {p}")
                    self.evaluate(self.imputed,plot=True)
                    self.compare_ts(self.imputed)

            res[label] = method_res

        return res

# analysis
    def find_max_period(self, start_time, end_time, param="co2",percent=10,periods=[30,60,90,120,150,180],methods=["MICE","missForest","ARIMA"],n_cv=3,verbose=False,plot=False,save=False):
        """
        Evaluates and compares imputation models on the consecutive missing observations datasets

        Parameters
        ----------
        param : str, default "co2"
            specifies which parameter to run the analysis for
        percent : int, default 10
            percents to consider - must have an accompanying data file
        periods : range or list, default [60,120,180,240,300,360]
            length in minutes/2 of the periods that have been removed
        verbose : boolean, default False
            whether or not to display output at each method iteration
        plot : boolean, default False
            whether to plot the imputed data

        Returns
        -------
        res : dictionary
            metric results for each method
        """
        res = {} # final results
        # missing data generator class
        missing_gen = TestData(pt=self.pt,data_dir=self.data_dir)
        missing_gen.restrict_data(start_time,end_time)
        self.set_base(missing_gen.data) # setting the base to the restricted data
        # defining the parameters 
        param_list = list(missing_gen.data.columns.values).copy()
        param_list.remove(param)
        self.param = param
        # getting results from each of the methods
        methods_dict = {"MICE":self.mice,"missForest":self.miss_forest,"ARIMA":self.arima}
        for label, method in methods_dict.items():
            if label in methods:
                if verbose:
                    print(label)
                method_res = {"Period":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[]}
                single_res = {"Period":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[]}
                for period in periods: # looping through the various period lengths
                    if verbose:
                        print(f"\t{period}")
                    raw_res = {"r2":[], "mae":[], "rmse":[], "ia":[]} 
                    for i in range(n_cv): # iterating 3 times for each period to get an average evaluation
                        iteration_start = datetime.now()
                        if verbose:
                            print("\t\tIteration:",i,end="")
                        # creating missing dataset
                        missing = missing_gen.remove_n_consecutive_with_other_missing(missing_gen.data,period,10,percent,param_list,param)
                        missing.set_index("timestamp",inplace=True)
                        self.set_missing(missing)
                        method(set_for_class=True)
                        if plot:
                            self.compare_ts(self.imputed)
                        try:
                            for metric, val in zip(raw_res.keys(),self.evaluate(self.imputed)):
                                raw_res[metric].append(val)
                        except Exception as e:
                            print("Error in evaluating imputation")
                            print(f"{param} - {period} - {percent}")
                            print(e)
                        iteration_end = datetime.now()
                        if verbose:
                            print(f" - {round((iteration_end - iteration_start).total_seconds(),1)}")

                    # averaging results from iterations
                    avg_res = list(pd.DataFrame(raw_res).mean().values)
                    avg_res.insert(0,period*2)
                    # appending average results to overall results
                    for metric, val in zip(method_res.keys(),avg_res):
                        method_res[metric].append(val)
                        single_res[metric] = val

                    # saving one iteration
                    if save:
                        with open(f"{self.data_dir}/data/interim/imputation/results/{label}-{period}-{param}.json", 'w') as fp:
                            json.dump(single_res, fp)
                        
                res[label] = method_res

        return res

    def optimize_rf(self,param="co2",percents=[10,20,30],n_estimators=[10,20,50,100],max_depths=[2,3,4],verbose=False,**kwargs):
        """
        Runs a gridsearch to determine the optimal RF model parameters

        Parameters
        ----------
        param : str, default "co2"
            which parameter to run the analysis for
        percents : list of int or float, default [10,20,30]
            percents to consider removing at random from the given parameter
        n_estimators : list of int, default [10,20,50,100]
            number of trees to consider in the random forest models
        max_depths : list of int, default [2,3,4]
            max depths to consider in the random forest models
        kwargs
            start_time : datetime
                to restrict the dataset
            end_time : datetime
                to restrict the dataset

        Returns
        -------
        res : DataFrame
            tabulated results from the gridsearch
        """
        self.param = param
        # missing data generator class
        missing_gen = TestData(pt=self.pt,data_dir=self.data_dir)
        # getting start and stop time from kwargs if available
        if "start_time" in kwargs.keys():
            start_time = kwargs["start_time"]
        else:
            start_time = missing_gen.data.index[0]
        if "end_time" in kwargs.keys():
            end_time = kwargs["end_time"]
        else:
            end_time = missing_gen.data.index[-1]
        missing_gen.restrict_data(start_time,end_time)

        self.set_base(missing_gen.data) # setting the base to the restricted data

        # defining the parameters 
        param_list = list(missing_gen.data.columns.values).copy()
        param_list.remove(param)
        # getting results for each model param
        res = {"Percent":[],"Estimators":[],"Depth":[],"Time to Execute":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[]}
        for percent in percents: # looping through the various period lengths
            for estimators in n_estimators:
                for max_depth in max_depths:
                    if verbose:
                        print(f"Percent: {percent} - Estimators: {estimators} - Depth: {max_depth}")
                    raw_res = {"r2":[], "mae":[], "rmse":[], "ia":[]} 
                    iteration_start = datetime.now()
                    for i in range(3): # iterating 3 times for each period to get an average evaluation
                        if verbose:
                            print(f"\tIteration {i}")
                        # creating missing dataset
                        some_missing = missing_gen.remove_at_random_all(missing_gen.data,percent=10,params=param_list) # removing fixed number from all but one param
                        missing = missing_gen.remove_at_random_all(some_missing,percent=percent,params=[param]) # removing the given percentage from the select param
                        self.set_missing(missing)

                        self.miss_forest(n_estimators=estimators,max_depth=max_depth,set_for_class=True)
                        
                        for metric, val in zip(raw_res.keys(),self.evaluate(self.imputed)):
                            raw_res[metric].append(val)

                    iteration_end = datetime.now()
                    t = (iteration_end - iteration_start).total_seconds()
                    if verbose:
                        print(f"\tTime to execute: {t} seconds")

                    # averaging results from iterations
                    avg_res = list(pd.DataFrame(raw_res).mean().values)
                    for i, val in enumerate([percent,estimators,max_depth,t]):
                        avg_res.insert(i,val)
                    # appending average results to overall results
                    for metric, val in zip(res.keys(),avg_res):
                        res[metric].append(val)

        return pd.DataFrame(res)
        
    def optimize_arima(self,param="co2",percent=15,periods=[15,30,60],ps=[2,3],ds=[0],qs=[1,2,3],verbose=False,**kwargs):
        """
        Runs a gridsearch to determine the optimal RF model parameters

        Parameters
        ----------
        param : str, default "co2"
            which parameter to run the analysis for
        percents : list of int or float, default [10,20,30]
            percents to consider removing at random from the given parameter
        ps : list of int, default [0,1,2]
            options for historic data points to consider
        ds : list of int, default [0,1,2]
            options for differencing order
        qs : list of int, default [0,1,2]
            options for moving average 
        kwargs
            start_time : datetime
                to restrict the dataset
            end_time : datetime
                to restrict the dataset

        Returns
        -------
        res : DataFrame
            tabulated results from the gridsearch
        """
        self.param = param
        # missing data generator class
        missing_gen = TestData(pt=self.pt,data_dir=self.data_dir)
        # getting start and stop time from kwargs if available
        if "start_time" in kwargs.keys():
            start_time = kwargs["start_time"]
        else:
            start_time = missing_gen.data.index[0]
        if "end_time" in kwargs.keys():
            end_time = kwargs["end_time"]
        else:
            end_time = missing_gen.data.index[-1]
        missing_gen.restrict_data(start_time,end_time)

        self.set_base(missing_gen.data) # setting the base to the restricted data

        # defining the parameters 
        param_list = list(missing_gen.data.columns.values).copy()
        param_list.remove(param)
        self.param = param
        # getting results for each model param
        res = {"Period":[],"P":[],"D":[],"Q":[],"Pearson Correlation":[],"MAE":[],"RMSE":[],"Index of Agreement":[],"AIC":[],"BIC":[],"HIC":[]}
        for period in periods: # looping through the various period lengths
            for p in ps:
                for d in ds:
                    for q in qs:
                        if verbose:
                            print(f"P:{p} - D:{d} - Q:{q}")
                        raw_res = {"r2":[], "mae":[], "rmse":[], "ia":[], "aic":[], "bic":[], "hic":[]} 
                        iteration_start = datetime.now()
                        for i in range(3): # iterating 3 times for each period to get an average evaluation
                            if verbose:
                                print(f"\tIteration {i}...")
                            # creating missing dataset
                            missing = missing_gen.remove_n_consecutive_with_other_missing(missing_gen.data,period,10,percent,param_list,param)
                            missing.set_index("timestamp",inplace=True)
                            self.set_missing(missing)

                            self.set_arima_order(order=(p,d,q))
                            aic, bic, hic = self.arima(set_for_class=True,return_metrics=True)
                            
                            for metric, val in zip(raw_res.keys(),self.evaluate(self.imputed)+(aic,bic,hic)):
                                raw_res[metric].append(val)

                        # averaging results from iterations
                        avg_res = list(pd.DataFrame(raw_res).mean().values)
                        for i, val in enumerate([period,p,d,q]):
                            avg_res.insert(i,val)
                        # appending average results to overall results
                        for metric, val in zip(res.keys(),avg_res):
                            res[metric].append(val)

                        iteration_end = datetime.now()
                        if verbose:
                            print(f"\tTime to execute: {(iteration_end - iteration_start).total_seconds()} seconds")

        return pd.DataFrame(res)

    def get_optimal_arima(self,res,sort_by="MAE"):
        """
        Determines the best ARIMA parameters from the DataFrame of results

        Parameters
        ----------
        res : DataFrame
            tabulated results from optimize_arima method
        sort_by : str, default "MAE"
            evaluation metric to sort by

        Returns
        -------
        optimal_params : DataFrame
            model parameters sorted by the order which produced the best results for all three percents
        """
        res_sorted = res.sort_values(["Period",sort_by],ascending=True)
        optimal_params = {"p":[],"d":[],"q":[],"rank":[]}
        for p in res["P"].unique():
            for d in res["D"].unique():
                for q in res["Q"].unique():
                    rank = 0 # initializing
                    for period in res["Period"].unique():
                        res_per_per = res_sorted[res_sorted["Period"] == period].reset_index()
                        val = res_per_per[(res_per_per["P"] == p) & (res_per_per["D"] == d) & (res_per_per["Q"] == q)].index[0]
                        rank += val
                    for ix, val in zip(["p","d","q","rank"],[p,d,q,rank]):
                        optimal_params[ix].append(val)

        optimal_params = pd.DataFrame(optimal_params).sort_values("rank").reset_index()
        print(f"P: {optimal_params.loc[0,'p']}, D: {optimal_params.loc[0,'d']}, Q: {optimal_params.loc[0,'q']}")
        print(f"Params at Lowest AIC:", res.sort_values("AIC").loc[0,"P"], res.sort_values("AIC").loc[0,"D"], res.sort_values("AIC").loc[0,"Q"])
        print(f"Params at Lowest BIC:", res.sort_values("BIC").loc[0,"P"], res.sort_values("BIC").loc[0,"D"], res.sort_values("BIC").loc[0,"Q"])
        return optimal_params

    def pacf_and_acf(self,param="co2",diff=False):
        """
        Uses the PACF and ACF plots to get the ideal ARMA parameters
        """
        if diff:
            temp = (self.base[param] - self.base[param].shift()).dropna()
        else:
            temp = self.base[param]
        plot_pacf(temp)
        plot_acf(temp)
# saving
    def save_res(self,res,save_dir="../",annot="_points"):
        """
        Save results from running points or periods at random

        Parameters
        ----------
        res : dictionary
            metric results for each method
        save_dir : str, default "../"
            location to data dir
        annot : str, default "_points"
            annotation to add to save filename

        Returns
        -------
        <void>
        """
        writer=pd.ExcelWriter(f"{save_dir}data/interim/imputation/{self.param}-results{annot}.xlsx") 
        for key in res.keys():
            pd.DataFrame(res[key]).to_excel(writer,sheet_name=key)

        writer.save()
        writer.close()