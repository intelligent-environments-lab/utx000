
from numpy.lib.scimath import sqrt
import pandas as pd
import numpy as np
from pandas.core import frame

# Iterative Imputer for MICE
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Evaluation Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Plotting
import matplotlib.pyplot as plt

class TestData:

    def __init__(self,pt,data_dir,params=["co2","pm2p5_mass","tvoc","temperature_c","rh"]):
        self.pt = pt
        self.data_dir = data_dir
        self.params = params

        self.data = self.import_example_data(self.pt,self.data_dir,self.params)

    def import_example_data(self,pt,data_dir="../",params=["co2","pm2p5_mass","tvoc","temperature_c","rh"]):
        """
        Imports example data
        
        Parameters
        ----------
        pt : str
            id of the participant to import for
        data_dir : str
            path to the "data" directory in the utx000 project
            
        Returns
        -------
        data : DataFrame
            exemplary ieq data from participant
        """
        
        try:
            data = pd.read_csv(f"{data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",
                            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        except FileNotFoundError:
            print("No filename for participant", pt)
            return pd.DataFrame()
        
        return data[params]

class Impute:

    def __init__(self,pt,data_dir,freq="2T",consecutive=False,prompt=False):
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
                self.load_data_consecutive_random(percent,period)
            else:
                self.load_data_random(percent)
        # Base Data
        self.base = pd.read_csv(f"{self.data_dir}data/interim/imputation/beacon-example-{pt}-ux_s20.csv",parse_dates=["timestamp"],
                                index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)

# methods for loading data
    def load_data_random(self,percent):
        """
        Loads in the randomly removed data
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_all-{self.param}-p{percent}-{self.pt}.csv",parse_dates=["timestamp"],
                                    index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)
        except FileNotFoundError as e:
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-{self.param}-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tparam:\t{self.param}\n\tpercent:\t{percent}")

    def load_data_consecutive_random(self,percent,period):
        """
        Loads in the randomly removed, consecutive data for class param
        """
        try:
            self.missing = pd.read_csv(f"{self.data_dir}data/interim/imputation/missing_data-random_periods_all-{self.param}-p{percent}-{period}mins-{self.pt}.csv",
                                    parse_dates=["timestamp"],index_col="timestamp",infer_datetime_format=True).asfreq(freq=self.freq)
        except FileNotFoundError:
            print("Could not find file: ",end="")
            print(f"missing_data-random_all-p{percent}-{self.pt}.csv")
            print(f"Check the parameters:\n\tparam:\t{self.param}\n\tpercent:\t{percent}\n\tperiod:\t{period}")

# methods for imputing
    def mice(self,estimator=None,set_for_class=False):
        """
        Imputes missing data with Mutiple Iterations of Chained Equations (MICE)
        
        Parameters
        ----------
        estimator : sklearn estimator, default None (Bayesion Ridge)
            estimator to use for IterativeImputer - see https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html?highlight=iterative%20imputer#sklearn.impute.IterativeImputer
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        imp = IterativeImputer(estimator=estimator,max_iter=30,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.mice_imputed = pd.DataFrame(imp.transform(self.missing),index=self.missing.index,columns=self.missing.columns)
        
        if set_for_class:
            self.set_imputed(self.mice_imputed)

    def miss_forest(self,set_for_class=False):
        """
        Imputes missing data with missForest

        Parameters
        ----------
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        imp = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10),max_iter=10,tol=1e-5,imputation_order="ascending")
        imp.fit(self.missing)
        self.rf_imputed = pd.DataFrame(imp.transform(self.missing),index=self.missing.index,columns=self.missing.columns)

        if set_for_class:
            self.set_imputed(self.rf_imputed)

    def arima(self,order=(2,1,2),set_for_class=False):
        """
        Imputes missing data with Auto-Regressive Integrated Moving Average 

        Parameters
        ----------
        order : tuple of three values
            order for (p, d, q)
        set_for_class : boolean, default False
            sets the imputed class variable to the results from this method

        Returns
        -------
        <void>
        """
        imp = ARIMA(self.missing[self.param], order=order, freq=self.freq)
        self.arima_imputed = self.missing.copy()
        self.arima_imputed[self.param] = imp.fit().predict()
        self.arima_imputed[self.param].replace(0,np.nanmean(self.arima_imputed[self.param]),inplace=True)
        if set_for_class:
            self.set_imputed(self.arima_imputed)

# setters
    def set_base(self,base):
        """
        Sets the class base data
        """
        self.base = base

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
        # placeholders for true/base and imputed values
        y_true = self.base.loc[self.missing[self.missing[self.param].isnull()].index.values,:][self.param].values
        y_pred = imputed.loc[self.missing[self.missing[self.param].isnull()].index.values,:][self.param].values
        # Metrics
        # -------
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

        if plot:
            _, ax = plt.subplots(figsize=(5,5))
            ax.scatter(y_true,y_pred,color="black",s=5)

            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

            ax.text(0.1,0.9,f"n={len(y_pred)}",transform=ax.transAxes)

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
        
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        
        plt.show()
        plt.close()

    def compare_methods(self,results,save=False,annot=""):
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
        fig, axes = plt.subplots(1,4,figsize=(18,4),gridspec_kw={"wspace":0.5})
        for metric, ax in zip(["Pearson Correlation","MAE","RMSE","Index of Agreement"],axes):
            for method,color in zip(results.keys(),["cornflowerblue","seagreen","firebrick"]):
                method_res = results[method]
                ax.plot(method_res["Percent"],method_res[metric],
                        lw=2,color=color,label=method)
                # Formatting
                # ----------
                # x-axis
                ax.set_xlim([0,50])
                ax.set_xticks(np.arange(0,55,5))
                # y-axis
                if metric in ["Pearson Correlation","Index of Agreement"]:
                    ax.set_ylim([0.3,1.0])
                else:
                    ax.set_ylim(bottom=0)
                # remainder
                ax.tick_params(labelsize=12)
                ax.set_title(metric,fontsize=16)
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
                
        # legend
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines,labels,loc="upper center",bbox_to_anchor=(0.5,0),frameon=False,ncol=3,fontsize=14)
        # common x-axis
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel("Percent Missing Data",labelpad=8,fontsize=14)
            
        if save:
            if annot != "":
                annot = "-" + annot
            plt.savefig(f"{self.data_dir}reports/figures/imputation/remove_at_random-method_metric_comparision{annot}.pdf",bbox_inches="tight")
        plt.show()
        plt.close()

# runners
    def run_at_random(self,param="co2",percents=[5,10,15,20,25,30,35,40,45,50]):
        """
        Evaluates and compares the imputation models

        Parameters
        ----------
        percents : range or list, default [5,10,15,20,25,30,35,40,45,50]
            percents to consider - must have an accompanying data file
        param : str, default "co2"
            specifies which parameter to run the analysis for

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
                self.load_data_random(percent=p)
                method(set_for_class=True)
                for metric, val in zip(method_res.keys(),(p,) + self.evaluate(self.imputed)):
                    method_res[metric].append(val)
                    
            res[label] = method_res

        return res

    def run_periods_at_random(self, param="co2", percents=[5,10,15,20,30,35,40,45,50],period=60):
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
                method(set_for_class=True)
                for metric, val in zip(method_res.keys(),(p,) + self.evaluate(self.imputed)):
                    method_res[metric].append(val)
                    
            res[label] = method_res

        return res
