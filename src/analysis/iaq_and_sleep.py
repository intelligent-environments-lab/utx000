# Ventilation 
# -----------
# Author: Hagen
# Date: 03/29/21
# Description: 

import sys
sys.path.append('../')

from src.visualization import visualize
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm

# factor analysis
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# stats
import math
from scipy import stats
from sklearn.metrics import matthews_corrcoef

class calculate():

    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data"):
        """
        Initializating method

        Parameters
        ----------
        study : str, default "utx000"
            study name
        study_suffix : str, default "ux_s20"
            study suffix ID
        data_dir : str, default "../../data"
            path to the "data" directory within the project

        Returns
        -------
        <void>
        """
        self.study = study
        self.suffix = study_suffix
        self.data_dir = data_dir

        # beacon data
        ## median nightly data
        ieq_raw = pd.read_csv(f"{self.data_dir}/processed/beacon_by_night-{self.suffix}.csv",parse_dates=["start_time","end_time",],infer_datetime_format=True)
        ieq_raw = ieq_raw[(ieq_raw["home"] == 1) | (ieq_raw["increasing_co2"] > 0.5)]
        ieq_raw = ieq_raw.groupby(["beacon","beiwe","redcap","start_time","end_time"]).median()
        ieq_raw = ieq_raw.add_suffix("_median")
        ieq_raw.reset_index(inplace=True)
        ieq_raw.dropna(subset=["tvoc_median","co_median","co2_median","pm2p5_mass_median"],how="all",inplace=True)
        ieq_raw.drop(["home_median","increasing_co2_median","ema_median","no2_median"],axis="columns",inplace=True)
        ieq_raw["end_date"] = pd.to_datetime(ieq_raw["end_time"].dt.date)
        self.ieq = ieq_raw.copy()
        ## ventilation estimates
        self.aer = pd.read_csv("../data/processed/beacon-ventilation.csv",parse_dates=["date"],infer_datetime_format=True)

        # fitbit data
        fb_sleep = pd.read_csv(f"{self.data_dir}/processed/fitbit-sleep_summary-{self.suffix}.csv",parse_dates=["start_time","end_time","start_date","end_date"],infer_datetime_format=True)#,index_col=["beiwe","start_time"])
        for stage in ["rem","nrem"]:
            fb_sleep[f"{stage}_percent"] = fb_sleep[f"{stage}_minutes"] / (fb_sleep["tst_fb"]*60)
        self.fb_sleep = fb_sleep
        
        # beiwe data
        beiwe_sleep = pd.read_csv(f"{self.data_dir}/processed/beiwe-morning_ema-{self.suffix}.csv",parse_dates=["timestamp"],infer_datetime_format=True)
        beiwe_sleep = self.remove_outlier(beiwe_sleep,"sol")
        beiwe_sleep["restful_binary"] = ["Positive" if score > 1 else "Negative" for score in beiwe_sleep["restful"]]
        beiwe_sleep["naw_binary"] = ["Low" if n < 2 else "High" for n in beiwe_sleep["naw"]]
        beiwe_sleep["end_date"] = pd.to_datetime(beiwe_sleep["timestamp"].dt.date)
        for likert in ["content","stress","energy","lonely","sad","restful"]:
            beiwe_sleep[likert] = pd.to_numeric(beiwe_sleep[likert])
        self.bw_sleep = beiwe_sleep
        
        # participant information
        pt_names = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='all')
        pt_names = pt_names[["beiwe","first","last","sex"]]
        pt_ids = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='beacon')
        pt_ids = pt_ids[['redcap','beiwe','beacon','lat','long','volume','roommates']] # keep their address locations
        self.info = pt_ids.merge(right=pt_names,on='beiwe')

    def summarize(self, df, by_id="beiwe"):
        """
        Summarizes available data per participant

        Parameters
        ----------
        df : DataFrame
            data
        by_id : str, default "beiwe"
            specifies which ID to use from ["beiwe","redcap","beacon","fitbit"]

        Returns
        -------
        <void>
        """  
        print("Number of Observations:", len(df))
        print("Number of Participants:", len(df[by_id].unique()))
        count = 0
        print("Observations per Participant:")
        for pt in df["beiwe"].unique():
            print(f"{pt}:\t{len(df[df[by_id] == pt])}")
            count += len(df[df[by_id] == pt])

    def plot_data_availability_heatmap(self, df, params=['tvoc','co2','co','pm2p5_mass','temperature_c'], agg_str = "", by_id="beacon", 
                                       df_filter='not', save=False, save_dir='../reports/figures/'):
        '''
        Plots a heatmap showing number of nights the various beacons measured for each sensor
        
        Parameters
        ----------
        df : DataFrame
            data with columns in var_list
        params : list of str, default ['tvoc','co2','co','pm2p5_mass','temperature_c']
            variables in df to consider
        agg_str : str, default ""
            string used in var_list if data were aggregated
        by_id : str, default "beacon"
            specifies which ID to use from ["beiwe","redcap","beacon","fitbit"]
        df_filter : str, default "not"
            the naming convention of the filtering applied to the beacon - used when saving
        save : boolean, default False
            whether to save the file
        save_dir : str, default '../reports/figures/'
            path to save the figure
        
        Returns
        -------
        df_count : DataFrame
            count dataframe used to generate the heatmap
        '''
        # filtering the dataframe to only include important vars
        columns_to_use = [column + agg_str for column in params] + [by_id,'start_time']
        df_filtered = df.copy()[columns_to_use]
        # dict to store the values
        data = {col: [] for col in columns_to_use if col != "start_time"}
        # looping through the dataframe to get the number of nights each beacon measured
        for bb in df_filtered[by_id].unique():
            df_by_bb_by_night = df_filtered[df_filtered[by_id] == bb]
            for var in columns_to_use:
                if var == "start_time":
                    pass
                elif var == by_id:
                    data[by_id].append(bb)
                else:
                    data[var].append(df_by_bb_by_night.count()[var])

        # formatting dataframe
        df_count = pd.DataFrame(data=data,index=data[by_id])
        df_count.drop(by_id,axis=1,inplace=True)
        df_count.sort_index(inplace=True)
        
        # plotting heatmap
        _, ax = plt.subplots(figsize=(12,5))
        sns.heatmap(df_count.sort_values(by=columns_to_use[0]).T,square=True,annot=True,fmt="d",linewidths=.5,cmap="Blues",vmin=0,vmax=70,cbar_kws={"shrink":0.5,"pad":0.02,"ticks":[0,10,20,30,40,50,60,70]},ax=ax)
        # reformatting figure labels
        tick_labels = []
        for param in params:
            tick_labels.append(visualize.get_label(param))
        ax.set_yticklabels(tick_labels,rotation=0)
        ax.set_xlabel(by_id.title())
        
        # saving and showing
        if save:
            plt.savefig(f'{save_dir}beacon_{df_filter}_filtered-data_availability-heatmap-ux_s20.pdf',bbox_inches='tight')

        plt.show()
        plt.close()
        
        return df_count

    def fill_na_with_mean(self,df,params=["tvoc","co","pm2p5_mass","co2","temperature_c"],agg_str="",by_id="beacon",verbose=False):
        """
        Fills NaN values with the mean value from that participant

        Parameters
        ----------
        df : DataFrame
            data
        params : list of str, default ['tvoc','co','co2','pm2p5_mass',"temperature_c"]
            variables in df to consider
        agg_str : str, default ""
            string used in var_list if data were aggregated
        by_id : str, default "beacon"
            specifies which ID to use from ["beiwe","redcap","beacon","fitbit"]
        verbose : boolean, default False
            extra output for debugging

        Returns
        -------
        df_filled : DataFrame
            original dataframe with missing values filled in
        """
        # including the aggregate to important columns
        columns_to_use = [column + agg_str for column in params]
        # initializing df to return
        df_filled = pd.DataFrame()
        # looping through all participants
        for bb in df[by_id].unique():
            df_bb = df[df[by_id] == bb] # pt-specific data
            # looping trhough all parameters
            for param in columns_to_use:
                nan_rows = df_bb[df_bb[param].isnull()]
                if len(nan_rows) > 0:
                    if verbose:
                        print(f"Number of NaN observations for {param}:", len(nan_rows))
                    df_bb[param] = df_bb[param].fillna(np.nanmean(df_bb[param])) # mean imputation
                    if verbose:
                        print("Number of NaN observations after filling:", len(df_bb[df_bb[param].isnull()]))
                        
                    if len(nan_rows) == len(df_bb[df_bb[param].isnull()]):
                        print(f"{by_id.title()} {bb} - {param.split('_')[0]}: There are likely no observations to fill in missing values with the mean.")
                
            df_filled = df_filled.append(df_bb)

        df_filled.dropna(inplace=True)
        return df_filled[columns_to_use]

    def scale_iaq(self, df):
        """
        Scales data with the Standard Scalar
        """

        return StandardScaler().fit_transform(df)
        
    def fa_adequacy_test(self,scaled_df,test="bartlett",verbose=False):
        """
        Performs an adequacy test on the scaled input data
        """
        if test == "bartlett":
            chi_sq, p = calculate_bartlett_sphericity(scaled_df)
            if verbose:
                print(f"Bartlett Test of Sphericity:\n\tChi:\t{chi_sq}\n\tp:\t{round(p,3)}")
            return chi_sq, p 
        elif test == "kmo":
            _, kmo_model=calculate_kmo(scaled_df)
            if verbose:
                print(f"Kaiser-Meyer-Olkin Tes:\n\tKMO:\t{kmo_model}")
            return kmo_model, 0
        else:
            print("Not a valid adequacy test")
            return np.nan, np.nan

    def run_factor_analysis(self, df, params=["tvoc","co","pm2p5_mass","co2","temperature_c"], agg_str = "", by_id="beacon", tests=["bartlett","kmo"],
                        n_factors=3,verbose=False):
        """
        Performs factor analysis 

        Parameters
        ----------
        df : DataFrame
            the data
        params : list of str, default ['tvoc','co','co2','pm2p5_mass',"temperature_c"]
            variables in df to consider
        agg_str : str, default ""
            string used in var_list if data were aggregated
        by_id : str, default "beacon"
            specifies which ID to use from ["beiwe","redcap","beacon","fitbit"]
        tests : list of str, default

        n_factors : int, default 3

        verbose : boolean, default False
            extra output for debugging

        Returns
        -------

        """
        # pre-processing
        df_filled = self.fill_na_with_mean(df,params=params,agg_str=agg_str,by_id=by_id)
        df_scaled = self.scale_iaq(df_filled)

        # getting test results
        for test in tests:
            _, _ = self.fa_adequacy_test(df_scaled,test=test,verbose=True)

        # creating factor analysis object and perform factor analysis
        fa = FactorAnalyzer(n_factors=n_factors,rotation=None,is_corr_matrix=False)
        fa.fit(df_scaled)

        evs = fa.get_eigenvalues()[0]
        if verbose:
            print("Eigenvalues:")
            print(evs)

        s = 0
        variances = []
        for ev in fa.get_eigenvalues()[0]:
            s += ev
            variances.append(s/len(fa.get_eigenvalues()[0]))
        if verbose:
            print("Variance:")
            print(variances)
        
        loadings = pd.DataFrame(fa.loadings_,index=[p.split("_")[0] for p in df_filled.columns],columns=[f"RC{i+1}" for i in np.arange(n_factors)])
        if verbose:
            print("Factor Loadings:")
            print(loadings)

        comms = fa.get_communalities()
        if verbose:
            print("Communalities:")
            print(comms)   

        spec_variances = fa.get_uniquenesses()
        if verbose:
            print("Specific Variances:")
            print(spec_variances)

        return evs, variances, loadings, comms, spec_variances

    def get_iaq_quality(self, iaq_val, threshold):
        """
        Determines if the iaq value is low or high based on the given threshold

        Parameters
        ----------
        iaq_val : float
            measured value of iaq parameter
        threshold : float
            limit to determine whether the measured value is low or high for that night

        Returns
        -------
        <result> : str
            one of [NaN, "low", or "high"]
        """
        if math.isnan(iaq_val):
            return np.nan
        elif iaq_val < threshold:
            return "low"
        else:
            return "high"

    def get_quality(self, df, threshold=0.35, above_label="Adequate", below_label="Inadequate", var_label="ach",
                                participant_based=False, by_id="beiwe"):
        """
        Determines the qualitative label for the variable based on the provided threshold or participant-based measurements.
        Defaults are set for ventilation rates, but this method is also be used for sleep quality.

        Parameters
        ----------
        df : DataFrame
            data
        threshold : float, default 0.35
            limit to determine whether the measured value is low or high
        above_label : str, default "Adequate"
            label to use for measurements above threshold
        below_label : str, default "Inadequate"
            label to use for measurements below threshold
        var_label : str, default "ach"
            column name corresponding to the variable that is to be binarized
        participant_based : boolean
            whether to base the quality off of the participants' measurement
        by_id : str, default "beiwe"
            ID used to loop through data

        Returns
        -------
        qualities : list of str
            list of qualitative ach rates
        """
        if participant_based == False:
            qualities =  [above_label if rate >= threshold else below_label for rate in df[var_label]]
        else:
            qualities = []
            for pt in df[by_id].unique():
                df_pt = df[df[by_id] == pt]
                qualities += [above_label if rate >= np.nanmean(df_pt[var_label]) else below_label for rate in df_pt[var_label]]
            
        return qualities

    def calculate_cramers_v(self,ct):
        """returns Cramers V from the given contingency table"""
        #create 2x2 table
        data = ct.values

        #Chi-squared test statistic, sample size, and minimum of rows and columns
        X2 = stats.chi2_contingency(data, correction=False)[0]
        n = np.sum(data)
        minDim = min(data.shape)-1

        #calculate Cramer's V 
        V = np.sqrt((X2/n) / minDim)

        #display Cramer's V
        return V

    def examine_binary_relationship(self, df_in, x_label="ach",x_threshold=0.35,y_labels=["tst_fb","efficiency","rem_percent"],y_thresholds=[7,85,0.2],
                                    participant_based=False):
        """
        Gets contigency table and analyzes relationship between binary measures

        Parameters
        ----------
        df_in : DataFrame
            data
        x_label : str, default "ach"
            label for the x variable
        x_threshold : float, default 0.35
            threshold to binarize x variable
        y_labels : list of str, default ["tst_fb","efficiency","rem_percent"]
            dependent variables to consider in df_in
        y_thresholds : list of float, default [7,0.85,0.2]
            thresholds to binary the dependent variables
        participant_based : boolean, default False
            whether to base the thresholds from participant measurements for both x and y variables

        Returns
        -------
        res : DataFrame
            contains contigency tables and Cramer's V scores
        """
        res = {"x":[],"y":[],"ct":[],"score":[]}
        df = df_in.copy()
        df[f"{x_label}_quality"] = self.get_quality(df,threshold=x_threshold,below_label="Low",above_label="High",participant_based=participant_based,var_label=x_label)
        for y, threshold in zip(y_labels,y_thresholds):
            df[f"{y}_quality"] = self.get_quality(df,threshold=threshold,below_label="Low",above_label="High",participant_based=participant_based,var_label=y)
            ct = pd.crosstab(columns=df[f"{x_label}_quality"],index=df[f"{y}_quality"])
            score = self.calculate_cramers_v(ct)
            for k, v in zip(res.keys(),[x_label,y,ct,score]):
                res[k].append(v)
            
        return pd.DataFrame(res)

    def remove_outlier(self, df, var, extreme="z",zscore=2.5):
        """
        Removes outlier observations based on the given variable

        Parameters
        ----------
        df : DataFrame
            data
        var : str
            specifies column in df to run outlier analysis on
        extreme : str, default "z"
            whether to run z-score ("z") or IQR (anything else)
        zscore : float, default 2.5 
            upper limit to use in zscore outlier identification

        Returns
        -------
        <df> : DataFrame
            original DataFrame with the outlier rows removed
        """
        if extreme == "z":
            df['z'] = abs(df[var] - df[var].mean()) / df[var].std(ddof=0)
            df["outlier"] = df['z'] > zscore
        else:
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            print("df")

            # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
            df["outlier"] = (df[var]<Q1-1.5*IQR) | (df[var]>Q3+1.5*IQR)
            
        df = df[df["outlier"] == False]
        df.drop(["z","outlier"],axis="columns",inplace=True)

        return df

    def plot_sleep_quality_violins(self, df, params=["tvoc","co","co2","pm2p5_mass","temperature_c"], limits=[200,4,1100,12,25.2],
                           sleep_metrics=["tst_fb","efficiency","rem2nrem"], iaq_metric="median",
                           save=False, annot="all", sleep_modality="fitbit", save_dir="../reports/figures/"):
        """
        Plots the main resulting figure looking at the distribution of the given sleep metrics for the iaq parameters

        Parameters
        ----------
        df : DataFrame
            iaq data summarized by night for each participant
        params : list of str, default ["tvoc","co","co2","pm2p5_mass","temperature_c"]
            iaq parameters to consider
        limits : list of float, default [200,4,1100,12,25.2]
            thresholds to consider for each parameter
        sleep_metrics : list of str, default ["tst_fb","efficiency","rem2nrem"],
            fitbit-based sleep metrics to consider
        iaq_metric : str, default "median"
            aggregate metric used for the iaq parameters per night
        save : boolean, default False
            whether to save the file
        annot : str, default "all"
            annotation to add to the save filename
        save_dir : str, default "../reports/figures/"
            path to save the figure
            
        Returns
        -------
        ttest_results : dict
            number of low, high observations, the means from these two distributions
        """
        # plot fontsizes
        legend_fs = 22
        tick_fs = 24
        label_fs = 26
        title_fs = 32

        df_to_plot = df.copy()

        # adding "low"/"high" column for each IAQ parameter
        thresholds = dict(zip(params,limits))
        for param, threshold in thresholds.items():
            df_to_plot[f"{param}_level"] = df_to_plot.apply(lambda x: self.get_iaq_quality(x[f"{param}_{iaq_metric}"],threshold),axis=1)

        # creating dictionary to store p-values
        ttest_results = {}
        # looping through sleep metrics
        _, axes = plt.subplots(len(sleep_metrics),1,figsize=(16,5*len(sleep_metrics)),sharex=True,gridspec_kw={"hspace":0.3})
        try:
            _ = len(axes)
        except TypeError:
            axes = [axes]
        for sleep_metric, title, ax in zip(sleep_metrics,["a","b","c","d"],axes):
            # expanding the df
            df_expanded = df_to_plot.melt(id_vars=[c for c in df_to_plot.columns if c.endswith("median") or c == sleep_metric],value_vars=[c for c in df_to_plot.columns if c.endswith("level")],value_name="level")
            
            # t_test
            # ------
            pvals = pd.DataFrame()
            for param in params:
                df = df_expanded[df_expanded["variable"] == f"{param}_level"]
                low_vals = df[df["level"] == "low"]
                high_vals = df[df["level"] == "high"]
                _, p = stats.ttest_ind(low_vals[sleep_metric],high_vals[sleep_metric], equal_var=True, nan_policy="omit")
                pvals = pvals.append(pd.DataFrame({"pollutant":[param],"low":[len(low_vals)],"high":[len(high_vals)],
                                                "mean_low":[np.nanmean(low_vals[sleep_metric])],"mean_high":np.nanmean(high_vals[sleep_metric]),"p_val":[p]}))

            ttest_results[sleep_metric.split("_")[0]] = pvals.set_index("pollutant")
            
            # plotting
            # --------
            sns.violinplot(x="variable",y=sleep_metric,hue="level",data=df_expanded,split=True,hue_order=["low","high"],palette={"low":"white","high":"#bf5700"},inner=None,cut=0,ax=ax)
            
            # x-axis
            ax.set_xticklabels([param.split("_")[0].upper().replace("O2","O$_2$").replace("2P5","$_{2.5}$").replace(" C","").replace("TEMPERATURE","T") for param in params],fontsize=tick_fs,va="top")
            ax.set_xlabel("")

            # y-axis
            plt.setp(ax.get_yticklabels(), ha="right", rotation=0, fontsize=tick_fs)
            # getting unit - need to have this so we don't put "()" for unitless metrics
            unit = visualize.get_units(sleep_metric)
            if unit == "":
                unit = ""
            else:
                unit = f"({unit})"
            ax.set_ylabel(f"{visualize.get_label(sleep_metric)} {unit}",fontsize=label_fs)

            # Modifying Remainder
            if sleep_metric == sleep_metrics[-1]:
                ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.18),frameon=False,ncol=2,fontsize=legend_fs,title_fontsize=tick_fs,title="Median Concentration")
            else:
                ax.get_legend().remove()
            for loc in ["top","right","bottom"]:
                ax.spines[loc].set_visible(False)
            ax.tick_params(axis=u'both', which=u'both',length=0)
            
            if sleep_metric != sleep_metrics[-1]:
                ax.axes.get_xaxis().set_visible(False)
                
            ax.text(-0.075,0.95,title,fontsize=title_fs,transform=ax.transAxes)

            # Annotating with p-values
            xlocs = ax.get_xticks()
            ax.text(ax.get_xlim()[0],ax.get_ylim()[1],"       p:",ha="center",va="bottom",fontsize=tick_fs)
            for xloc, p, n_low, n_high in zip(xlocs,ttest_results[sleep_metric.split("_")[0]]["p_val"],ttest_results[sleep_metric.split("_")[0]]["low"],ttest_results[sleep_metric.split("_")[0]]["high"]):
                weight="bold" if p < 0.05 else "normal"
                val = round(p,3) if p > 0.001 else "<0.001"
                ax.text(xloc,ax.get_ylim()[1],val,fontsize=tick_fs,ha="center",va="bottom",weight=weight)
                if sleep_metric == sleep_metrics[-1]:
                    ax.text(xloc,-0.12,f"{n_low} ",fontsize=tick_fs,ha="right",va="top")
                    ax.text(xloc,-0.12,f" {n_high}",fontsize=tick_fs,ha="left",va="top")
                
        ax.text(0,-0.22,"n:",fontsize=tick_fs,transform = ax.transAxes)

        if save:
            plt.savefig(f'{save_dir}/beacon_{sleep_modality}/beacon-{sleep_modality}-{iaq_metric}_profile-{annot}-ux_s20.pdf',bbox_inches="tight")

        plt.show()
        plt.close()
        
        return ttest_results

class device_sleep(calculate):

    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data"):
        super().__init__(study=study, study_suffix=study_suffix, data_dir=data_dir)
        # IAQ and Sleep
        data = self.ieq.merge(right=self.fb_sleep,on=["start_time","end_time","beiwe","redcap","beacon"],how='inner', indicator=False)
        self.sleep_and_iaq_data = data[[column for column in data.columns if not column.endswith("delta")]]  
        # Ventilation and Sleep
        aer_and_fb = self.aer.merge(right=self.fb_sleep,left_on=["beiwe","beacon","date"],right_on=["beiwe","beacon","end_date"])
        aer_and_fb = aer_and_fb[aer_and_fb["method"].isin(["ss","decay_2"])]
        aer_and_fb.replace({"decay_2":"decay"},inplace=True)    
        self.sleep_and_aer_data = aer_and_fb

    def plot_ventilation_violin(self, df_in, yvar="tst_fb", binary_var="ventilation_quality", threshold=0.35, zero_label="Inadequate", one_label="Adequate", participant_based=False,
                           save=False, save_dir="../reports/figures/beacon_fitbit"):
        """
        Plots violin plots of sleep metric distributions for binary outcomes
        
        Parameters
        ----------
        df_in : DataFrame
            data
        yvar : str, default "tst_fb"
            variable that will be split into two distributions
        binary_var : str, default "ventilation_quality"
            variable used to define distributions
        zero_label : str, default "Inadequate"
            specifies which qualitative specification maps to a 0
        one_label : str, default "Adequate"
            specifies which qualitative specification maps to a 1
        save : boolean, default False
            whether to save the file
        save_dir : str, default "../reports/figures/beacon_fitbit/"
            path to save the figure

        Returns
        -------
        res : DataFrame

        """
        df = df_in.copy()
        df[binary_var] = self.get_quality(df,threshold=threshold,above_label=one_label,below_label=zero_label,participant_based=participant_based)

        # t-Test
        # ------
        low_vals = df[df[binary_var] == zero_label]
        high_vals = df[df[binary_var] == one_label]
        _, p = stats.ttest_ind(low_vals[yvar],high_vals[yvar], equal_var=True, nan_policy="omit")
        res = pd.DataFrame({"parameter":[yvar],"low":[len(low_vals)],"high":[len(high_vals)],
                                                "mean_low":[np.nanmean(low_vals[yvar])],"mean_high":np.nanmean(high_vals[yvar]),"p_val":[p]})

        # Plotting
        # --------
        # plot fontsizes
        legend_fs = 22
        tick_fs = 24
        label_fs = 26

        _, ax = plt.subplots(figsize=(6,6))
        df["target"] = yvar
        sns.violinplot(x="target",y=yvar,hue=binary_var,split=True,hue_order=[zero_label,one_label],palette={zero_label:"#bf5700",one_label:"white"},data=df,ax=ax,cut=0,inner=None)
        for loc in ["right","top","bottom"]:
            ax.spines[loc].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_legend().remove()
        if visualize.get_units(yvar) == "":
            unit = ""
        else:
            unit = " (" + visualize.get_units(yvar) + ")"
        ax.set_ylabel(visualize.get_label(yvar) + unit,fontsize=label_fs)
        plt.yticks(fontsize=tick_fs)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1), title=binary_var.replace("_"," ").title(),ncol=1,frameon=False,title_fontsize=label_fs,fontsize=legend_fs)

        # Annotating with p-values
        xloc = ax.get_xticks()
        weight="bold" if p < 0.05 else "normal"
        ax.text(xloc-0.1,ax.get_ylim()[1],"p:  ",fontsize=tick_fs,ha="center",va="bottom")
        ax.text(xloc,ax.get_ylim()[1],f"{round(p,2)}",fontsize=tick_fs,ha="center",va="bottom",weight=weight)
        ax.text(xloc,ax.get_ylim()[0],f"{len(low_vals)} ",fontsize=tick_fs,ha="right",va="top")
        ax.text(xloc,ax.get_ylim()[0],f" {len(high_vals)}",fontsize=tick_fs,ha="left",va="top")

        if save:
            plt.savefig(f"{save_dir}/{yvar}-{binary_var}-violin.pdf", bbox_inches="tight")
            
        plt.show()
        plt.close()

        return res.set_index("Parameter")

class report_sleep(calculate):
    
    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data"):
        super().__init__(study=study, study_suffix=study_suffix, data_dir=data_dir)
        # IAQ and Sleep
        self.sleep_and_iaq_data = self.ieq.merge(right=self.bw_sleep, on=["end_date","beiwe","redcap","beacon"])
        # Ventilation and Sleeo
        aer_and_bw = self.aer.merge(right=self.bw_sleep, left_on=["beiwe","date"], right_on=["beiwe","end_date"])
        self.sleep_and_aer_data = aer_and_bw

    def plot_iaq_violin(self, df_in, sleep_metric="restful", targets=["tvoc","co","co2","pm2p5_mass","temperature_c"], hues=["Negative","Positive"],
                        save=False, save_dir="../reports/figures/beacon_ema",**kwargs):
        """
        Plots violin plots of concentration distributions for positive and negative sleep metrics scores
        
        Parameters
        ----------
        df_in : DataFrame
            data
        sleep_metric : str, default "restful"
            label for sleep metric
        targets : list of str, default ["tvoc"","co","co2","pm2p5_mass","temperature_c"]
            iaq parameters to consider
        hues : list of str, default ["Negative","Positive"]
            labels for the binary sleep metric
        save : boolean, default False
            whether to save the file
        save_dir : str, default "../reports/figures/beacon_ema"
            path to save the figure

        Returns
        -------
        pvals : DataFrame
            summary and results from the t-Test
        """
        # plot font sizes
        tick_fs = 24
        label_fs = 26

        df = df_in.copy()
        pvals = pd.DataFrame()
        _, axes = plt.subplots(2,3,figsize=(15,8),gridspec_kw={"wspace":0.5,"hspace":0.3})
        for target, ax in zip(targets,axes.flat):
            target = target+"_median"
            df["target"] = target

            # t-Test
            # ------
            low_vals = df[df[f"{sleep_metric}_binary"] == hues[0]]
            high_vals = df[df[f"{sleep_metric}_binary"] == hues[1]]
            _, p = stats.ttest_ind(low_vals[target],high_vals[target], equal_var=True, nan_policy="omit")
            pvals = pvals.append(pd.DataFrame({"parameter":[target],"low":[len(low_vals)],"high":[len(high_vals)],
                                                "mean_low":[np.nanmean(low_vals[target])],"mean_high":np.nanmean(high_vals[target]),"p_val":[p]}))

            # Plotting
            # --------
            sns.violinplot(x="target",y=target,hue=f"{sleep_metric}_binary",hue_order=hues,split=True,palette={hues[0]:"white",hues[1]:"seagreen"},data=df,ax=ax,cut=0,inner=None,)
            # x
            ax.get_xaxis().set_visible(False)
            # y
            ax.set_ylabel(f"{visualize.get_label(target.split('_median')[0])} ({visualize.get_units(target.split('_median')[0])})",fontsize=label_fs)
            ax.tick_params(labelsize=tick_fs)
            if "ticks" in kwargs.keys():
                ax.set_yticks(kwargs["ticks"][target.split('_median')[0]])
            # remainder
            for loc in ["right","top","bottom"]:
                ax.spines[loc].set_visible(False)
            ax.get_legend().remove()
            
            # Annotating with p-values
            xloc = ax.get_xticks()
            weight="bold" if p < 0.05 else "normal"
            val = round(p,3) if p > 0.001 else "< 0.001"
            ax.text(xloc,ax.get_ylim()[1],f"p: {val}",fontsize=tick_fs,ha="center",va="bottom",weight=weight)
            ax.text(0.5,-0,f"{len(low_vals)} ",fontsize=tick_fs,ha="right",va="top",transform = ax.transAxes)
            ax.text(0.5,-0,f" {len(high_vals)}",fontsize=tick_fs,ha="left",va="top",transform = ax.transAxes)

        if save:
            plt.savefig(f"{save_dir}/all_ieq-binary_{sleep_metric}-violin.pdf", bbox_inches="tight")

        plt.show()
        plt.close()
        
        return pvals.set_index("parameter")
    
def main():
    pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='iaq_and_sleep.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()