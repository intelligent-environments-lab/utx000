
# General
import os
import logging

# Data Science
import pandas as pd
import numpy as np

# Stats
from scipy import stats
from scipy.stats import linregress

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class fitbit_sleep():

    def __init__(self, path_to_data="../../data", path_to_figures="../../reports/figures"):
        # initializing read and save locations
        self.path_to_data = path_to_data
        self.path_to_figures = path_to_figures
        # getting data
        self.data = pd.read_csv(f"{self.path_to_data}/processed/fitbit_fitbit-daily_activity_and_sleep-ux_s20.csv",parse_dates=["date","start_date","end_date","start_time","end_time"])
        # cleaning 
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
        df_to_return = df_to_return[df_to_return["nrem_percent"] > 0]
        return df_to_return

    def add_guidelines_met(self, df, activity_level="moderately", threshold=150):
        """
        Adds column to see if a guideline was met based on an activity level and threshold

        Inputs:
        - df:
        - activity level:
        - threshold:

        Returns void; column should have been added to dataframe
        """
        df[f"{activity_level}_met"] = df[f"{activity_level}_weekly"] >= threshold

    def plot_violins(self, df, checks=["moderately_met","vigorously_met","combined_met"], sleep_metrics=["rem_percent","wake_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"], annotate=False, save_dir="../reports/figures", save=False):
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
            ax.set_xlabel("Recommendation Basis")
            # y-axis
            ax.set_ylabel(sleep_metric.replace("_"," ").upper()) 
            # remainder
            for loc in ["top","right"]:
                    ax.spines[loc].set_visible(False)
            ax.legend(title=f"Weekly Recommendation Met?",loc="lower center",bbox_to_anchor=(1.1,0.5),frameon=False)

            # Annotating with p-values
            if annotate:
                pvals = []
                ns = []
                for check in checks:
                    temp = df_expanded[df_expanded["condition"] == check]
                    low_vals = temp[temp["met"] == "False"]
                    high_vals = temp[temp["met"] == "True"]
                    _, p = stats.ttest_ind(low_vals[sleep_metric],high_vals[sleep_metric], equal_var=True)
                    pvals.append(p)
                    ns.append([len(low_vals),len(high_vals)])

                xlocs = ax.get_xticks()
                ax.text(ax.get_xlim()[0],ax.get_ylim()[1],"          p:",ha="center",va="bottom",fontsize=12)
                for xloc, p, n in zip(xlocs,pvals, ns):
                    weight="bold" if p < 0.05 else "normal"
                    ax.text(xloc,ax.get_ylim()[1],f"{round(p,3)} ({n[0]},{n[1]})",fontsize=12,ha="center",va="bottom",weight=weight)

            if save:
                plt.savefig(f"{save_dir}/fitbit_summary/weekly_activity_recommendations-{sleep_metric}-violin.png")

            plt.show()
            plt.close()

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
                    df_by_pt_no_zeros["slope"] = np.polyfit(df_by_pt_no_zeros[f"{activity_level}_active_minutes"],df_by_pt_no_zeros[sleep_metric],deg=order)[0]
                    df_by_pt_no_zeros["r"] = abs(np.polyfit(df_by_pt_no_zeros[f"{activity_level}_active_minutes"],df_by_pt_no_zeros[sleep_metric],deg=order,full=True)[1][0])
                    df_cleaned = df_cleaned.append(df_by_pt_no_zeros)
                
        return df_cleaned.sort_values("r",ascending=True)

    def plot_indvidual_responses(self, df, sleep_metrics = ["rem_percent","wake_percent","nrem_percent","rem2nrem_percent","efficiency","tst_fb"], save_dir="../reports/figures",save=False):
        """

        """
        for sleep_metric in sleep_metrics:
            for activity_level in ["moderately","combined","vigorously"]:
                df_to_plot = self.get_cleaned_activity_df(df,activity_level,sleep_metric)
                fig, axes = plt.subplots(4,11,figsize=(22,8),sharey=True,sharex=True,gridspec_kw={"wspace":0,"hspace":0})
                for pt, ax in zip(df_to_plot["redcap"].unique(),axes.flat):
                    df_to_plot_by_pt = df_to_plot[df_to_plot["redcap"] == pt]
                    sns.regplot(x=f"{activity_level}_active_minutes",y=sleep_metric,data=df_to_plot_by_pt,ci=None,truncate=True,order=1,
                                scatter_kws={"s": 20, "alpha":0.7,"color":"cornflowerblue"},
                                line_kws={"linewidth":1,"color":"black"},ax=ax)
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_title(int(pt),y=1.0)
                        
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    
                fig.text(0.1,0.5,sleep_metric.replace("_"," ").replace("2",":").title(), ha='center', va='center', rotation='vertical',size=16)
                fig.text(0.5, 0.07, f"{activity_level.title()} Active Minutes per Day", ha='center', va='center',size=16)

                axes[3,0].get_xaxis().set_visible(True)
                axes[3,0].get_yaxis().set_visible(True)
                plt.show()
                plt.close()

    def run_aggregate_analysis(self, thresholds={"moderately":150,"vigorously":75,"combined":150}):
        """

        """
        # getting weekly activity data and checking to see if guidelines are met
        df = self.get_active_minutes_per_week(self.data)
        for level, threshold in thresholds.items():
            self.add_guidelines_met(df,level,threshold)

        print(df)


    def run_individual_analysis(self):
        """

        """
        df = self.get_active_minutes_per_week(self.data)

def main():
    activity_and_fitbit_sleep = fitbit_sleep()
    activity_and_fitbit_sleep.run_aggregate_analysis()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='activity_and_sleep.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()