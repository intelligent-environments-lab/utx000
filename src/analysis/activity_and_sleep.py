
# General
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Data Science
import pandas as pd
import numpy as np

# Stats
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import r2_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class base():

    def __init__(self):
        pass

    def get_active_minutes_per_week(self, raw_data, index_col="date", id_col="redcap"):
        """
        Using a rolling sum to get the moderate, vigorous, and combined active minutes per week
        """
        df = raw_data.copy()
        df.set_index(index_col,inplace=True)
        df_to_return = pd.DataFrame()
        # looping through participants to include weekly data
        for pt in df[id_col].unique():
            act_by_pt = df[df[id_col] == pt]
            if len(act_by_pt) >= 7: # need at least a week's worth of data (might not be consecutive still)
                # getting date range of data and reindexing to include missing days
                dt = pd.date_range(min(act_by_pt.index), max(act_by_pt.index))
                act_by_pt = act_by_pt.reindex(dt, fill_value=0)
                # creating new activity columns and dropping remaining
                act_by_pt["moderately_active_minutes"] = act_by_pt["fairly_active_minutes"]
                act_by_pt["vigorously_active_minutes"] = act_by_pt["very_active_minutes"]
                act_by_pt["combined_active_minutes"] = act_by_pt["fairly_active_minutes"] + act_by_pt["very_active_minutes"]*2
                act_by_pt.drop(["sedentary_minutes","lightly_active_minutes","fairly_active_minutes","very_active_minutes"],axis="columns",inplace=True)
                # getting rolling sums
                for level in ["moderately","vigorously","combined"]:
                    act_by_pt[f"{level}_weekly"] = act_by_pt[f"{level}_active_minutes"].rolling(7,7).sum()
                    # backfilling first entries with the mean value
                    act_by_pt[f"{level}_weekly"].fillna(value=np.nanmean(act_by_pt[f"{level}_weekly"]))
                # appending to final dataframe
                df_to_return = df_to_return.append(act_by_pt)

        # removing unobserved days by using dummy column
        df_to_return = df_to_return[df_to_return["steps"] > 0]
        return df_to_return

    def add_guidelines_met(self, df, activity_level="moderately", threshold=150):
        """Adds column to see if a guideline was met based on an activity level and threshold"""
        df[f"{activity_level}_met"] = df[f"{activity_level}_weekly"] >= threshold

    def plot_violins(self, df, checks=["moderately_met","vigorously_met","combined_met"], sleep_metrics=["rem_percent","wake_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"], annotate=False, save_dir="../reports/figures", save=False, save_annotation=""):
        """

        """
        df_expanded = df.melt(id_vars=sleep_metrics,value_vars=checks,var_name="condition",value_name="met")
        df_expanded.replace(False,"False",inplace=True)
        df_expanded.replace(True,"True",inplace=True)
        for sleep_metric in sleep_metrics:
            _, ax = plt.subplots(figsize=(4*len(checks),4))
            sns.violinplot(x="condition",y=sleep_metric,hue="met",data=df_expanded,split=True,hue_order=["False","True"],palette={"False":"white","True":"seagreen"},inner="quartile",cut=0,ax=ax)
            # x-axis
            ax.set_xticklabels([check.split("_met")[0].replace("_","-").title()+" Active Minutes" for check in checks])
            ax.set_xlabel("Guidline Basis")
            # y-axis
            ax.set_ylabel(sleep_metric.replace("_"," ").upper()) 
            # remainder
            for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
            ax.legend(title=f"Guideline Met?",ncol=1,frameon=False)

            # Annotating with p-values
            if annotate:
                pvals = []
                ns = []
                for check in checks:
                    temp = df_expanded[df_expanded["condition"] == check]
                    p, n = self.calculate_pvalue(temp,"met",sleep_metric, ("False","True"))
                    pvals.append(p)
                    ns.append(n)

                xlocs = ax.get_xticks()
                ax.text(ax.get_xlim()[0],ax.get_ylim()[1],"          p:",ha="center",va="bottom",fontsize=12)
                for xloc, p, n in zip(xlocs,pvals, ns):
                    weight="bold" if p < 0.05 else "normal"
                    ax.text(xloc,ax.get_ylim()[1],f"{round(p,3)} ({n[0]},{n[1]})",fontsize=12,ha="center",va="bottom",weight=weight)

            if save:
                plt.savefig(f"{save_dir}/weekly_activity_recommendations-{sleep_metric}{save_annotation}-violin.png")

            plt.show()
            plt.close()

    def calculate_pvalue(self, df, split_col, target_col, split_labels=(False,True), equal_var=True):
        """Gets p-values and number of observatiosn between distributions"""
        low_vals = df[df[split_col] == split_labels[0]]
        high_vals = df[df[split_col] == split_labels[1]]
        _, p = stats.ttest_ind(low_vals[target_col],high_vals[target_col], equal_var=equal_var)
        return p, [len(low_vals),len(high_vals)]

class fitbit_sleep(base):

    def __init__(self, path_to_data="../../data", path_to_figures="../../reports/figures/fitbit_summary"):
        # initializing read and save locations
        self.path_to_data = path_to_data
        self.path_to_figures = path_to_figures
        # getting and cleaning Fitbit data
        self.data = pd.read_csv(f"{self.path_to_data}/processed/fitbit_fitbit-daily_activity_and_sleep-ux_s20.csv",parse_dates=["date","start_date","end_date","start_time","end_time"])
        self.data.drop(["water_logged","food_calories_logged","fat","bmr","start_date","end_date","timestamp","end_time","start_time","beiwe","beacon"],axis="columns",inplace=True)
        self.data = self.data[self.data["efficiency"] > 75]
        self.add_sleep_percentage(self.data)

    def add_sleep_percentage(self, df):
        """Converts and drops minute columns into percentage columns"""
        for column in df.columns:
            # looking for sleep columns only
            if column.endswith("minutes") and column not in ["sedentary_minutes","lightly_active_minutes","fairly_active_minutes","very_active_minutes"]:
                variable = column.split("_")[0]
                # converting to percent and dropping
                df[variable+"_percent"] = df[column] / (df["tst_fb"]*60) * 100
                df.drop(column,axis="columns",inplace=True)

        # getting rem:nrem percent
        df["rem2nrem_percent"] = df["rem_percent"] / df["nrem_percent"]

    def get_pvalues(self, df, checks=["moderately_met","vigorously_met","combined_met"], sleep_metrics=["rem_percent","wake_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"]):
        """

        """
        df_expanded = df.melt(id_vars=sleep_metrics,value_vars=checks,var_name="condition",value_name="met")
        df_expanded.replace(False,"False",inplace=True)
        df_expanded.replace(True,"True",inplace=True)
        d = {"target":[],"activity_guideline":[],"n_unmet":[],"n_met":[],"p":[]}
        for sleep_metric in sleep_metrics:
            for check in checks:
                temp = df_expanded[df_expanded["condition"] == check]
                p, n = self.calculate_pvalue(temp,"met",sleep_metric, ("False","True"))
                d["target"].append(sleep_metric)
                d["activity_guideline"].append(check)
                d["n_unmet"] .append(n[0])
                d["n_met"].append(n[1])
                d["p"].append(p)
                
        return pd.DataFrame(data=d)

    def run_regression(self, x, y, order=1):
        """Gets sum of residuals for the order of polynomial fit"""
        fit = np.polyfit(x,y,deg=order,full=True)
        coeff = fit[0][0] # leading coefficient
        const = fit[0][-1] # constant
        resid = fit[1][0] # sum of residuals
        return coeff, const, resid

    def get_cleaned_activity_df(self, df, activity_level, sleep_metric, order=1, id_col="redcap", min_points=3):
        """
        Removes participants from dataframe and orders according to a given column
        """
        df_cleaned = pd.DataFrame()
        for pt in df[id_col].unique():
            df_by_pt = df[df[id_col] == pt]
            if len(df_by_pt) >= 7:
                df_by_pt_no_zeros = df_by_pt[df_by_pt[f"{activity_level}_active_minutes"] > 0]
                if len(df_by_pt_no_zeros) >= min_points:
                    df_by_pt_no_zeros["n"] = len(df_by_pt_no_zeros)
                    coeff, const, resid = self.run_regression(df_by_pt_no_zeros[f"{activity_level}_active_minutes"], df_by_pt_no_zeros[sleep_metric], order=order)
                    df_by_pt_no_zeros["slope"] = coeff
                    df_by_pt_no_zeros["r"] = resid / len(df_by_pt_no_zeros)
                    df_by_pt_no_zeros["constant"] = const
                    df_cleaned = df_cleaned.append(df_by_pt_no_zeros)
                
        return df_cleaned.sort_values("r",ascending=True)

    def plot_scatter(self, df, levels=["combined","vigorously"], sleep_metrics=["rem_percent","wake_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"], save=False, save_dir="../reports/figures/fitbit_summary"):
    
        for sleep_metric in sleep_metrics:
            _, axes = plt.subplots(1,len(levels),figsize=(8*len(levels),6),sharey=True,gridspec_kw={"wspace":0})
            for level, ax in zip(levels,axes):
                sns.scatterplot(x=f"{level}_weekly",y=sleep_metric,hue=f"{level}_met",palette=["black","seagreen"],data=df,ax=ax)
                # x-axis
                ax.set_xlabel(level.title())
                ax.set_xlim(left=0)
                # remainder
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
            # y-axis
            axes[0].set_ylabel(sleep_metric.replace("_"," ").upper()) 
            
            if save:
                plt.savefig(f"{save_dir}/{level}_active_minutes-{sleep_metric}-scatter.png",bbox_inches="tight")
                
            plt.show()
            plt.close()

    def plot_indvidual_responses(self, df, levels=["combined","vigorously"], sleep_metrics=["wake_percent","rem_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"], save_dir="../reports/figures/fitbit_summary", save=False, show=False):
        """

        """
        for sleep_metric in sleep_metrics:
            for level in levels:
                df_to_plot = self.get_cleaned_activity_df(df,level,sleep_metric)
                fig, axes = plt.subplots(4,11,figsize=(22,8),sharey=True,sharex=True,gridspec_kw={"wspace":0,"hspace":0})
                for pt, ax in zip(df_to_plot["redcap"].unique(),axes.flat):
                    df_to_plot_by_pt = df_to_plot[df_to_plot["redcap"] == pt]
                    sns.regplot(x=f"{level}_active_minutes",y=sleep_metric,data=df_to_plot_by_pt,ci=None,truncate=True,order=1,
                                scatter=False,line_kws={"linewidth":1,"color":"black"},ax=ax)
                    try:
                        sns.scatterplot(x=f"{level}_active_minutes",y=sleep_metric,hue=f"{level}_met",palette=["black","seagreen"],data=df_to_plot_by_pt,ax=ax,legend=False)
                    except ValueError:
                        sns.scatterplot(x=f"{level}_active_minutes",y=sleep_metric,color="black",data=df_to_plot_by_pt,ax=ax,legend=False)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title(int(pt),y=1.0)
                        
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    
                fig.text(0.1,0.5,sleep_metric.replace("_"," ").replace("2",":").title(), ha='center', va='center', rotation='vertical',size=16)
                fig.text(0.5, 0.07, f"{level.title()} Active Minutes per Day", ha='center', va='center',size=16)

                axes[3,0].get_xaxis().set_visible(True)
                axes[3,0].get_yaxis().set_visible(True)

                if save:
                    plt.savefig(f"{save_dir}/{level}-{sleep_metric}-individual_scatter.png",bbox_inches="tight")
                if show:
                    plt.show()
                plt.close()

    def run(self, thresholds={"moderately":150,"vigorously":75,"combined":150}):
        """

        """
        # getting weekly activity data and checking to see if guidelines are met
        df = self.get_active_minutes_per_week(self.data)
        for level, threshold in thresholds.items():
            self.add_guidelines_met(df,level,threshold)

        #self.plot_violins(df, annotate=True, save_dir=self.path_to_figures, save=True)
        #print(self.get_pvalues(df))
        self.plot_indvidual_responses(df, save_dir=self.path_to_figures, save=True)

class ema_sleep(base):

    def __init__(self):
        pass

    def plot_strip(self, df, levels=["combined","vigorously"], save=False, save_dir="../reports/figures/ema_fitbit"):
        for level in levels:
            _, axes = plt.subplots(1,2,figsize=(12,4),sharey=False, gridspec_kw={'wspace':0.1})
            for met, ax in zip([False, True], axes.flat):
                temp = df[df[f"{level}_met"] == met]
                sns.stripplot(x="restful", y=f"{level}_weekly", data=temp, jitter=.2, alpha=0.7, ax=ax)
                labels = ax.get_xticklabels()
                c = temp["restful"].value_counts().sort_index()
                new_labels = []
                for label, count in zip(labels,c):
                    new_labels.append(f"{label.get_text()} ({count})")
                ax.set_xticklabels(new_labels)
                ax.set_xlabel("Restful Score")
                ax.set_ylim(bottom=0)
                for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)

            axes[0].set_ylabel(f"{level.title()} Active Minutes (Weekly)")
            axes[1].set_ylabel("")

            if save:
                plt.savefig(f"{save_dir}/{level}_active_minutes-restful-stripplot.png",bbox_inches="tight")
            plt.show()
            plt.close()

def main(): 
    activity_and_fitbit_sleep = fitbit_sleep()
    activity_and_fitbit_sleep.run(thresholds={"moderately":300,"vigorously":150,"combined":300})

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='activity_and_sleep.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()