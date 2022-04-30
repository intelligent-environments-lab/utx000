# IAQ and Mood
# -----------
# Author: Hagen
# Date: 02/209/22
# Description: 

import sys

from attr import assoc
sys.path.append('../')

# user-created libraries
from src.visualization import visualize
from src.analysis import aqi

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns

class Process():

    def __init__(self, data_dir="../data", study_suffix="ux_s20") -> None:
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
        ema_all : DataFrame
            morning and evening EMAs filled out during the entire study
        iaq : DataFrame
            all available beacon IAQ data
        """
        self.data_dir = data_dir
        self.suffix = study_suffix

        # EMA data
        # --------
        ## At Home Responses
        ema = pd.read_csv(f"{self.data_dir}/processed/beiwe-ema_at_home_v2-{self.suffix}.csv",
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        for column in ema.columns:
            if column != "beiwe":
                ema[column] = pd.to_numeric(ema[column]) # sometimes the data are not numeric

        ## Including extra columns
        ema["minutes_at_home"] = ema["time_at_home"] / 60
        ema["DoW"] = ema.index.strftime("%a")

        self.ema = ema

        ## All Responses
        self.ema_all = pd.read_csv(f"{self.data_dir}/processed/beiwe-daily_ema-{self.suffix}.csv",
            index_col="date",parse_dates=["timestamp_morning","timestamp_evening","date"],infer_datetime_format=True)

        # IAQ data
        iaq = pd.read_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv',
            index_col="timestamp",parse_dates=["timestamp"],infer_datetime_format=True)
        iaq.drop(["beacon","redcap","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm10_mass","no2","lux"],axis=1,inplace=True)
        for column in iaq.columns:
            if column != "beiwe":
                iaq[column] = pd.to_numeric(iaq[column]) # might as well

        self.iaq = iaq

    def process(self,add_discontent=True,binarize_ema=True,scale=False):
        """
        Processes data for further analyses

        Parameters
        ----------
        add_discontent : boolean, default True
            whether to add discontent column
        binarize_ema : boolean, default False
            whether to binarize ema responses
        scale : boolean, default False
            whether to scale the data

        Updates
        -------
        ema : DataFrame
            numeric responses to the EMAs completed at home
        """
        if add_discontent:
            self.add_discontent()

        if binarize_ema:
            self.binarize_ema()

        if scale:
            self.standardize_ema()

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

        Updates
        -------
        ema : DataFrame
            adds binary column for each mood
        """
        for mood in moods:
            if mood in ["content","discontent","energy"]:
                self.ema[f"{mood}_binary"] = [0 if score < 2 else 1 for score in self.ema[mood]]
            else:
                self.ema[f"{mood}_binary"] = [0 if score == 0 else 1 for score in self.ema[mood]]

    def standardize_ema(self,moods=["content","discontent","stress","lonely","sad","energy"]):
        """
        Standardizes mood responses

        Parameters
        ----------
        moods : list of str, default ["content","discontent","stress","lonely","sad","energy"]
            moods in ema to consider

        Updates
        -------
        ema : DataFrame
            adds scaled column for each mood
        """
        scaler = StandardScaler()
        
        for mood in moods:
            self.ema[f"{mood}_scaled"] = scaler.fit_transform(self.ema[mood].values.reshape(-1, 1)) 

    def get_na(self,moods=["discontent","stress","lonely","sad"]):
        """
        Gets a Negative Affect score from the provided moods

        Parameters
        ----------
        moods : list of str, default ["discontent","stress","lonely","sad"]
            moods to use for score

        Updates
        -------
        ema_and_iaq : DataFrame
            adds in na column
        """
        self.ema["na"] = self.ema[[col for col in self.ema.columns if col in moods]].sum(axis=1)

    def label_iaq(self,value,threshold):
        """
        Labels binary IAQ measurements. Really just needed to handle NaNs
        """
        if value == np.nan:
            return np.nan
        elif value > threshold:
            return 0
        elif value <= threshold:
            return 1
        else:
            return np.nan

class Explore(Process):
    """
    Exploration on single datasets and simplistic analyses
    """

    def __init__(self, data_dir="../data", study_suffix="ux_s20") -> None:
        super().__init__(data_dir, study_suffix)

    def get_mood_distribution(self, moods=["discontent","stress","lonely","sad"], binary=False, plot=False):
        """
        Parameters
        ----------
        moods : list-like, default ["content","stress","lonely","sad"]
            Strings of the moods to consider - must be columns in df_in
        binary : boolean, default False
            whether to use binary values or not
        plot : boolean, default False
            whether or not to output the histograms of the scores
            
        Returns
        -------
        df : DataFrame
            
        """
        if binary:
            scores= [0,1]
        else:
            scores = [0,1,2,3]

        response_summary = {score: [] for score in scores}
        response_summary["mood"] = []
        for mood in moods:
            response_summary["mood"].append(mood)
            for score in scores:
                response_summary[score].append(len(self.ema[self.ema[mood] == score]))
                
        response_summary_df = pd.DataFrame(response_summary).set_index("mood")

        if plot:
            _, axes = plt.subplots(1,len(moods),figsize=(len(moods)*4,3),sharey="row",sharex="col")
            for mood, ax in zip(moods,axes):
                ax.bar(response_summary_df.columns,response_summary_df.loc[mood,:],edgecolor="black",color="lightgray")
                #for score, height in zip(dist.index,dist.values/dist.sum()):
                    #ax.text(score,height+0.05,round(height*100,1),ha="center")
                #ax.set_ylim([0,1])
                    
                ax.set_xlabel(mood.title(),fontsize=12)

                # appending results to output
                #res[mood].append(dist.values)

                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
                
            plt.show()
            plt.close()
            
        return response_summary_df

    def plot_mood_on_time(self,moods=["discontent","stress","lonely","sad"],interval=15,max_t=600):
        """
        Plots aggregated mood scores submitted at different lengths after arriving home

        Parameters
        ----------
        moods : list of str, default ["discontent","stress","lonely","sad"]
            moods in ema to consider
        interval : int, default 15
            time interval, in minutes, between each submission group
        max_t : int, default 600
            max time, in minutes, a participant submitted an EMA after arriving home

        Returns
        -------

        """
        _, axes = plt.subplots(len(moods),1,figsize=(18,3*len(moods)),sharex=True)
        for mood, ax in zip(moods,axes):
            scores = []
            ns = []
            for cutoff1, cutoff2 in zip(np.arange(0,max_t,interval),np.arange(interval,max_t+interval,interval)):
                ema_by_cutoff = self.ema[(self.ema["minutes_at_home"] >= cutoff1) & (self.ema["minutes_at_home"] <= cutoff2)]
                scores.append(ema_by_cutoff.mean(axis=0)[mood])
                ns.append(len(ema_by_cutoff))

            sc = ax.scatter(np.arange(interval,max_t+interval,interval),scores,s=100,c=ns,cmap=plt.cm.get_cmap("coolwarm_r", 8),vmin=0,vmax=80,edgecolor="black")
            ax.set_title(mood.title(),fontsize=22,pad=0)

            ax.set_xticks(np.arange(0,max_t+interval,interval))

            ax.set_ylim([-0.5,3.5])
            ax.set_yticks(np.arange(0,4,1))
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            ax.tick_params(labelsize=18)
                
        ax.set_xlabel("Minutes since arriving home",fontsize=20)
        cbar = plt.colorbar(sc,ax=axes.ravel().tolist(),shrink=0.75)
        cbar.ax.set_title("EMAs",fontsize=18)
        cbar.ax.tick_params(labelsize=18)
        plt.show()
        plt.close()

    def plot_mood_on_dow(self,scale=False):
        """
        Plots the mean aggregated mood scores for each day of the week (DoW)

        Parameters
        ----------
        scale : boolean, default False
            whether to use scaled values or not
        """
        moods = ["discontent","stress","lonely","sad"] # hardcoded because of the 2x2 structure
        if scale:
            moods = [f"{mood}_scaled" for mood in moods]
        ema_copy = self.ema.copy()
        ema_copy["DoW"] = ema_copy.index.strftime("%a")
        n_df = ema_copy.groupby("DoW").count()
        ema_copy_dow = ema_copy.groupby("DoW").mean()
        ema_copy_dow["n"] = n_df["beiwe"]
        ema_copy_dow = ema_copy_dow.reindex(index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,gridspec_kw={"hspace":0,"wspace":0})
        for mood, color, ax in zip(moods, ["seagreen","goldenrod","firebrick","cornflowerblue"],axes.flat):
            ax.scatter(ema_copy_dow.index,ema_copy_dow[mood],s=50,c=color)
            # x-axis

            # y-axis
            if scale:
                pass
            else:
                ax.set_ylim([0,1.25])
                ax.set_yticks(np.arange(0,1.25,0.25))
            # remainder
            ax.tick_params(labelsize=14)
            ax.set_title(mood.title(), fontsize=16)

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel("Mean Aggregated Mood Score",fontsize=18,labelpad=15)

        plt.show()
        plt.close()

        return ema_copy_dow

    def categorize_time_at_home(self,bin_edges=[60,120,180,240,300,360]):
        """
        Creates categorical column for time at home
        """
        minutes_at_home_cat = []
        for time_at_home in self.ema["minutes_at_home"]:
            if time_at_home > bin_edges[-1]: 
                minutes_at_home_cat.append(f">{bin_edges[-1]}")
            else:
                for edge in bin_edges:
                    if time_at_home <= edge:
                        minutes_at_home_cat.append(f"<{edge}")
                        break

        self.ema["minutes_at_home_cat"] = minutes_at_home_cat

    def conduct_mood_anova(self, mood, group_label, run_tukey=True):
        """
        Conducts one-way analysis of variance for a given mood
        
        Parameters
        ----------
        mood : str
            mood to consider
        group_label : str
            grouping column
        run_tukey : boolean, True
            whether to run Tukey HSD between groups

        Returns
        -------

        """
        # Splitting data into groups
        ema_mood = self.ema[[mood,group_label]]
        group_data = []
        for group in ema_mood[group_label].unique():
            group_data.append(ema_mood[ema_mood[group_label] == group][mood].values)
        
        # ANOVA
        fvalue, pvalue = stats.f_oneway(*group_data)
        print(f"{mood.upper()}\nANOVA p-value: {round(pvalue,3)}")

        #Tukey HSD
        if run_tukey:
            res = pairwise_tukeyhsd(ema_mood[mood],ema_mood[group_label])
            print(res)

        return fvalue, pvalue, res
    
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
    
    def determine_mood_associations(self, moods=["discontent","stress","lonely","sad"], binary=False):
        """
        Displays associations as determined by Cramer's V between all combinations of mood.

        Parameters
        ----------
        moods : list of str, default ["discontent","stress","lonely","sad"]
            moods to consider from the EMA
        binary : boolean, default False
            whether to use binary responses or not

        Returns
        -------
        associations : DataFrame
            tabularized results
        """
        associations = {"mood1":[],"mood2":[],"score":[]}
        for mood1 in moods:
            for mood2 in moods:
                if mood1 != mood2:
                    if binary:
                        V = self.calculate_cramers_v(pd.crosstab(self.ema[f"{mood1}_binary"],self.ema[f"{mood2}_binary"]))
                    else:
                        V = self.calculate_cramers_v(pd.crosstab(self.ema[mood1],self.ema[mood2]))
                        
                    for key, value in zip(associations.keys(),[mood1,mood2,V]):
                        associations[key].append(value)

        return pd.DataFrame(associations)

class Analyze(Process):

    def __init__(self, data_dir="../data", study_suffix="ux_s20") -> None:
        super().__init__(data_dir, study_suffix)

    def binarize_iaq(self,thresholds=None,comparator="mean"):
        """
        Gets the binary encoding of the vars in pollutants for each participant
        
        Parameters
        ----------
        thresholds : dict or str, default None
            thresholds to binarize the corresponding IAQ parameters
            None - use default values defined in this method
            dict - keys corresponding to IAQ parameters and values are the thresholds
            str - one of ["median","mean"] meaning to use the participant's specific median/mean value as the threshold
        comparator : str, default "mean"
            which IAQ summary statistic to compare the the threshold to

        Updates
        -------     
        ema_and_iaq : DataFrame
            Includes a binary IAQ column where 0 is bad/above and 1 is good/below
        """ 
        updated_ema_and_iaq = pd.DataFrame()

        for pt in self.ema_and_iaq["beiwe"].unique():
            # pt-specific data
            iaq_pt = self.iaq[self.iaq["beiwe"] == pt]
            ema_and_iaq_pt = self.ema_and_iaq[self.ema_and_iaq["beiwe"] == pt]
            # getting thresholds in order
            if thresholds == None:
                threshold_pt = {"co2": 1100, "co": 4, "tvoc": 200, "pm2p5_mass": 12, "temperature_c": 25.2, "rh": 60}
            elif thresholds == "median":
                threshold_pt = {param: np.nanmedian(iaq_pt[param]) for param in ["co2","co","tvoc","pm2p5_mass","temperature_c","rh"]}
                comparator = "median"
            elif thresholds == "mean":
                threshold_pt = {param: np.nanmean(iaq_pt[param]) for param in ["co2","co","tvoc","pm2p5_mass","temperature_c","rh"]}
            else: # already in dictionary form
                threshold_pt = thresholds

            for param, threshold in threshold_pt.items():
                try:
                    ema_and_iaq_pt[f"{param}_binary"] = ema_and_iaq_pt.apply(lambda x: self.label_iaq(x[f"{param}_{comparator}"],threshold), axis="columns")
                except KeyError:
                    pass # IAQ parameter not included - used primarily for co

            updated_ema_and_iaq = updated_ema_and_iaq.append(ema_and_iaq_pt)

        self.ema_and_iaq = updated_ema_and_iaq

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

        self.ema_and_iaq = ema_and_iaq.dropna(subset=["co2_mean","pm2p5_mass_mean","temperature_c_mean","rh_mean"],how="all")
        self.iaq_prior = iaq_prior.dropna(subset=["co2","pm2p5_mass","temperature_c","rh"],how="all")

    def plot_mood_violin_per_iaq(self,iaq_params=["co2","pm2p5_mass","tvoc","temperature_c"],summary_stat="mean",moods=["discontent","stress","lonely","sad","energy"],label="",save=False):
        """
        Plots the distributions of IAQ parameters for binary moods

        Parameters
        ----------
        iaq_params : list of str, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            IAQ parameters to include for visualization
        summary_stat : str, default "mean"
            summary statistic to use
        """
        legend_fs = 22
        tick_fs = 24
        label_fs = 26
        # creating dictionary to store p-values
        ttest_results = {}
        _, axes = plt.subplots(len(iaq_params),1,figsize=(4*len(moods),4*len(iaq_params)),sharex=True)
        for iaq_param, ax in zip(iaq_params,axes.flat):
            df_expanded = self.ema_and_iaq.melt(id_vars=[c for c in self.ema_and_iaq.columns if c.endswith(summary_stat)],value_vars=moods)
            g = sns.violinplot(x="variable",y=f"{iaq_param}_{summary_stat}",hue="value",data=df_expanded,
                        split=True,inner=None,hue_order=[0,1],palette={0:"white",1:"#bf5700"},cut=0,ax=ax,legend_out=False)
            # x-axis
            ax.set_xticklabels([mood.title() for mood in moods],fontsize=tick_fs)
            ax.set_xlabel("")
            # y-axis
            plt.setp(ax.get_yticklabels(), ha="right", rotation=0, fontsize=tick_fs)
            ax.set_ylabel(visualize.get_label(iaq_param),fontsize=label_fs)
            # remainder
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            if iaq_param == iaq_params[-1]:
                ax.legend(handles=g.get_children(),labels=["Low","High"],loc="upper center",bbox_to_anchor=(0.5,-0.075),frameon=False,ncol=2,fontsize=legend_fs,title_fontsize=tick_fs,title="Mood Score")
            else:
                ax.get_legend().remove()
            
            pvals = pd.DataFrame()
            for mood in moods:
                df = df_expanded[df_expanded["variable"] == mood].dropna()
                low_vals = df[df["value"] == 0]
                high_vals = df[df["value"] == 1]
                #print(f"Number of high:\t{len(high_vals)}\nNumber of low:\t{len(low_vals)}")
                _, p = stats.ttest_ind(low_vals[f"{iaq_param}_{summary_stat}"],high_vals[f"{iaq_param}_{summary_stat}"], equal_var=True, nan_policy="omit")
                pvals = pvals.append(pd.DataFrame({"mood":[mood.title()],"low":[len(low_vals)],"high":[len(high_vals)],
                                                "mean_low":[np.nanmean(low_vals[f"{iaq_param}_{summary_stat}"])],"mean_high":np.nanmean(high_vals[f"{iaq_param}_{summary_stat}"]),"p":[p]}))

            ttest_results[iaq_param] = pvals.set_index("mood")

            # Annotating with p-values
            xlocs = ax.get_xticks()
            ax.text(ax.get_xlim()[0],ax.get_ylim()[1],"          p:",ha="center",va="bottom",fontsize=tick_fs)
            for xloc, p in zip(xlocs,ttest_results[iaq_param]["p"]):
                weight="bold" if p < 0.1 else "normal"
                sty="italic" if p < 0.05 else "normal"
                val = round(p,3) if p > 0.001 else "< 0.001"
                ax.text(xloc,ax.get_ylim()[1],val,fontsize=tick_fs,ha="center",va="bottom",weight=weight,fontstyle=sty)
            
        if save:
            plt.savefig(f"../reports/figures/beiwe-beacon-mood_bi{'_'+label}-ieq_{summary_stat}{'_'+label}-violin-ux_s20.pdf",bbox_inches="tight")
        
        plt.show()
        plt.close()
        
        return ttest_results

    def plot_correlation_matrix(self, params=["co2","pm2p5_mass","tvoc","temperature_c",], metric="mean", save=False, annot="partially_filtered"):
        """
        Plots correlation matrix between variables
        
        Parameters
        ----------
        params : list of str, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            names of columns to include when summarizing
        metric : str, default "mean"
            summary stat for the IAQ parameters to consider
        save : boolean, default False
            whether to save or not
        annot : str, default None
            information to include in the filename when saving

        Creates
        -------
        correlation_matrix : NumPy Array
            correlation matrix between the given parameters
        """
        # getting dataframe in order
        params_with_metric = [param+f"_{metric}" for param in params]
        df = self.ema_and_iaq[params_with_metric]
        df.columns = [visualize.get_label(col) for col in params]

        corr = df.corr()
        corr = round(corr,2)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        _, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, mask=mask, 
                        vmin=-1, vmax=1, center=0, 
                        cmap=sns.diverging_palette(20, 220, n=200),cbar_kws={'ticks':[-1,-0.5,0,0.5,1],"pad":-0.07,"shrink":0.8,"anchor":(0.0,0.0)},fmt=".2f",
                        square=True,linewidths=1,annot=True,annot_kws={"size":12},ax=ax)
        # colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.outline.set_color('black')
        cbar.outline.set_linewidth(0.5)
        
        yticklabels = ax.get_yticklabels()
        yticklabels[0] = ' '
        ax.set_yticklabels(yticklabels,rotation=0,ha='right',fontsize=14)

        xticklabels = ax.get_xticklabels()
        xticklabels[-1] = ' '
        ax.set_xticklabels(xticklabels,rotation=0,ha='center',fontsize=14)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        if save:
            if annot:
                plt.savefig(f'../data/processed/beacon-{annot}-correlation_matrix-{self.suffix}.csv')
            else:
                plt.savefig(f'../data/processed/beacon-correlation_matrix-{self.suffix}.csv')
                
        plt.show()
        plt.close()

        self.correlation_matrix = corr

    def determine_mood_and_binary_iaq(self, moods=["discontent","stress","lonely","sad"], iaq_params=["co2","pm2p5_mass","tvoc","temperature_c"], binary_mood=False):
        """
        Displays associations as determined by Cramer's V between all combinations of mood.

        Parameters
        ----------
        moods : list of str, default ["discontent","stress","lonely","sad"]
            moods to consider from the EMA
        iaq_params : list of str, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            IAQ parameters to consider
        binary_mood : boolean, default False
            whether to use binary mood responses or not

        Returns
        -------
        associations : DataFrame
            tabularized results
        """
        associations = {"mood":[],"iaq_param":[],"score":[]}
        for mood in moods:
            for iaq_param in iaq_params:
                if binary_mood:
                    V = self.calculate_cramers_v(pd.crosstab(self.ema_and_iaq[f"{mood}_binary"],self.ema_and_iaq[f"{iaq_param}_binary"]))
                else:
                    V = self.calculate_cramers_v(pd.crosstab(self.ema_and_iaq[mood],self.ema_and_iaq[f"{iaq_param}_binary"]))
                    
                for key, value in zip(associations.keys(),[mood,iaq_param,V]):
                    associations[key].append(value)

        return pd.DataFrame(associations)

    def get_aggregate_aqi(self,iaq_params=["co2","pm2p5_mass","tvoc","temperature_c"],summary_stat="mean",use_binary=False):
        """
        Gets aggregate AQI metric

        Parameters
        ----------
        iaq_params : list of str, default ["co2","pm2p5_mass","tvoc","temperature_c"]
            IAQ parameters to use for aggregate aqi
        summary_stat : str, default "mean"
            summary stat to calculate AQI for each IAQ parameters
        use_binary : boolean, default False
            use binary IAQ rather than summarized concentrations

        Updates
        -------
        ema_and_iaq : DataFrame
            Adds in AQI for each IAQ parameter and creates and aggregate AQI
        """
        if use_binary:
             self.ema_and_iaq["agg_aqi"] = self.ema_and_iaq[[col for col in self.ema_and_iaq.columns if col.split("_binary")[0] in iaq_params]].sum(axis=1)
        else:
            function_map = {"co2":aqi.co2,"pm2p5_mass":aqi.pm2p5_mass,"tvoc":aqi.tvoc,"temperature_c":aqi.trh}
            for iaq_param in iaq_params:
                f = function_map[iaq_param]
                aqis = []
                for measurement, rh_measurement in zip(self.ema_and_iaq[f"{iaq_param}_{summary_stat}"],self.ema_and_iaq[f"rh_{summary_stat}"]):
                    if iaq_param in ["co2","tvoc"]:
                        aqis.append(f(measurement))
                    elif iaq_param == "pm2p5_mass":
                        aqis.append(f(measurement,indoor=True))
                    elif iaq_param == "temperature_c":
                        aqis.append(f(measurement,rh_measurement/100,data_dir=self.data_dir))

                self.ema_and_iaq[f"{iaq_param}_aqi"] = aqis

            self.ema_and_iaq["agg_aqi"] = self.ema_and_iaq[[col for col in self.ema_and_iaq.columns if col.split("_aqi")[0] in iaq_params]].sum(axis=1)

    def plot_aqi_vs_mood(self,mood="discontent"):
        """
        Box plots of AQI versus the discrete and binary mood scores
        """
        fig, axes = plt.subplots(1,2,figsize=(12,4),sharey=True,gridspec_kw={"wspace":0.1,'width_ratios': [4, 1]})
        sns.violinplot(x=mood,y="agg_aqi",data=self.ema_and_iaq,
            cut=0,inner=None,ax=axes[0])
        # x-axis
        axes[0].set_xlabel("Discrete",fontsize=14)
        # y-axis
        axes[0].set_ylabel("AQI",fontsize=14)
        # remainders
        axes[0].tick_params(labelsize=12)
        # remainder
        for loc in ["top","right"]:
            axes[0].spines[loc].set_visible(False)
        axes[0].set_title(mood.title(),fontsize=18)

        sns.violinplot(x=f"{mood}_binary",y="agg_aqi",data=self.ema_and_iaq,
            cut=0,inner=None,split=True,ax=axes[1])
         # x-axis
        axes[1].set_xlabel("Binary",fontsize=14)
        # y-axis
        axes[1].set_ylabel("",fontsize=14)
        # remainder
        axes[1].tick_params(labelsize=12)
        for loc in ["top","right"]:
            axes[1].spines[loc].set_visible(False)

        plt.show()
        plt.close()

    def mean_mood_difference(self,data,mood="stress"):
        """
        Calculates the mean mood difference between the given data and overall data
        """
        mean_response_pt = self.ema.copy()[["beiwe","discontent","stress","lonely","sad","energy"]].groupby("beiwe").mean()
        combined = data.merge(mean_response_pt,on="beiwe",how="left",suffixes=["","_mean"])
        combined[f"{mood}_diff"] = combined[mood] - combined[f"{mood}_mean"]
        summarized = combined.groupby("beiwe").mean()
        return combined, np.nanmean(summarized[f"{mood}_diff"])

    def compare_mean_mood_scores(self,ieq_param="co2",moods=["discontent","stress","sad","lonely","energy"]):
        """
        Compares the mood scores between the extreme and non-extreme cases
        """
        try:
            df = self.ema_and_iaq.copy()
        except AttributeError as e:
            print(e)
            return pd.DataFrame()

        res = {"mean_low":[],"mean_high":[],"p":[]}
        
        high_ieq = df[df[f"{ieq_param}_binary"] == 1]
        low_ieq = df[df[f"{ieq_param}_binary"] == 0]
        print(f"High: \t{len(high_ieq)}\nLow:\t{len(low_ieq)}")
        
        for mood in moods:
            high_mean = np.nanmean(high_ieq[mood])
            low_mean = np.nanmean(low_ieq[mood])
            high_std = np.nanstd(high_ieq[mood])
            low_std = np.nanstd(low_ieq[mood])
            u, p = stats.mannwhitneyu(low_ieq[mood].values,high_ieq[mood].values)
            if p < 0.05:
                p = f"{round(p,3)}*"
            elif p < 0.1:
                p = f"{round(p,3)}**"
            else:
                p = f"{round(p,3)}"
            for key, val in zip(res.keys(),[(round(low_mean,2),round(low_std,2)),(round(high_mean,2),round(high_std,2)),p]):
                if len(val) == 2:
                    #res[key].append(f"{val[0]} ({val[1]})")
                    res[key].append(val[0])
                else:
                    res[key].append(val)
                    
        Moods = [mood.title() for mood in moods]
        return pd.DataFrame(data=res,index=Moods)
    
    def compare_difference_mood_scores(self,ieq_param="co2",moods=["discontent","stress","sad","lonely","energy"]):
        """
        Compares the mood scores between the extreme and non-extreme cases

        Parameters
        ----------
        """
        try:
            df = self.ema_and_iaq.copy()
        except AttributeError as e:
            print(e)
            return pd.DataFrame()

        res = {"mean_low":[],"mean_high":[],"p":[]}
        
        high_ieq = df[df[f"{ieq_param}_binary"] == 1]
        low_ieq = df[df[f"{ieq_param}_binary"] == 0]
        print(f"High: \t{len(high_ieq)}\nLow:\t{len(low_ieq)}")
        
        for mood in moods:
            high_data, high_mean = self.mean_mood_difference(high_ieq,mood=mood)
            low_data, low_mean = self.mean_mood_difference(low_ieq,mood=mood)
            _, p = stats.mannwhitneyu(low_data[f"{mood}_diff"].values,high_data[f"{mood}_diff"].values)
            if p < 0.05:
                p = f"{round(p,3)}*"
            elif p < 0.1:
                p = f"{round(p,3)}**"
            else:
                p = f"{round(p,3)}"
            for key, val in zip(res.keys(),[round(low_mean,2),round(high_mean,2),p]):
                res[key].append(val)
                    
        Moods = [mood.title() for mood in moods]
        return pd.DataFrame(data=res,index=Moods)