import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import sys

from sklearn.utils import resample
sys.path.append('../')

from src.visualization import visualize

class Compare:

    def __init__(self,study1,study2,beacons1,beacons2,
        params=["tvoc","co","co2","pm2p5_mass"],resample_rate=1,remove_outliers=False,data_dir="../") -> None:
        """
        Parameters
        ----------

        """
        self.data_dir = data_dir
        self.params = params
        self.resample_rate = resample_rate

        self.study1 = study1
        self.bb1 = beacons1
        self.data1 = self.get_data(self.study1,self.bb1,params=self.params,resample_rate=self.resample_rate,remove_outliers=remove_outliers)
        
        self.study2 = study2
        self.bb2 = beacons2
        self.data2 = self.get_data(self.study2,self.bb2,params=self.params,resample_rate=self.resample_rate,remove_outliers=remove_outliers)
        
    def resample_data(self,df,resample_rate=1):
        """
        Resamples data per participant

        Parameters
        ----------

        Returns
        -------

        """
        df_resampled = pd.DataFrame()
        for bb in df["beacon"].unique():
            df_bb = df[df["beacon"] == bb]
            df_bb_resampled = df_bb.resample(f"{resample_rate}T").mean()
            df_bb_resampled["beacon"] = bb
            df_resampled = df_resampled.append(df_bb_resampled)
            
        return df_resampled

    def get_data(self, study_suffix, beacon_list, params, resample_rate=1, remove_outliers=False):
        """
        Sets beacon data from the provided study
        
        Parameters
        ----------
        study_suffix : str
        
        beacon_list : list of int
        
        params : list of str, default ["tvoc","co","co2","pm2p5_mass"]
        
        resample_rate : int, default 1
            resampling time in minutes
        remove_outliers : boolean, default False
            whether to remove outliers outside zscore of 2.5

        Returns
        -------
        temp_restricted_filtered_resampled : DataFrame
            
        """
        try:
            temp = pd.read_csv(f"../data/processed/beacon-{study_suffix}.csv",index_col=0,parse_dates=["timestamp"],infer_datetime_format=True)
        except FileNotFoundError:
            print("No beacon found from study", study_suffix)
            return
        
        temp_restricted = temp[temp["beacon"].isin(beacon_list)]
        temp_restricted_filtered = temp_restricted[params+["beacon"]]
        temp_restricted_filtered_resampled = self.resample_data(temp_restricted_filtered, resample_rate=resample_rate)
        if remove_outliers:
            for param in params:
                temp_restricted_filtered_resampled['z'] = abs(temp_restricted_filtered_resampled[param] - temp_restricted_filtered_resampled[param].mean()) / temp_restricted_filtered_resampled[param].std(ddof=0)
                temp_restricted_filtered_resampled.loc[temp_restricted_filtered_resampled['z'] > 2.5, param] = np.nan
                
            temp_restricted_filtered_resampled.drop(["z"],axis=1,inplace=True)
            
        temp_restricted_filtered.dropna(how="all",subset=params,inplace=True)
        
        return temp_restricted_filtered_resampled

    # Visuals
    # -------
    def plot_kde(self, df1=None, df2=None, param="co2", study1="UTx000", study2="Ambassador Families"):
        """
        Plots the KDE for each parameter
        """
        if df1 is None:
            df1 = self.data1.copy()
        
        if df2 is None:
            df2 = self.data2.copy()
            
        _, ax = plt.subplots(figsize=(16,4))
        s1_kde = sns.kdeplot(param,cut=0,data=df1,ax=ax,
                linewidth=2,color="seagreen",label=study1)
        s2_kde = sns.kdeplot(param,cut=0,data=df2,ax=ax,
                linewidth=2,color="cornflowerblue",label=study2)
        
        # x-axis
        ax.set_xlabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})",fontsize=16)
        # y-axis
        ax.set_ylabel("Density",fontsize=16)
        # remainder
        ax.tick_params(labelsize=14)
        plt.legend(frameon=False,fontsize=14)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        
        plt.show()
        plt.close()
    
    def plot_boxplot_comparison(self,df1,df2,params=["tvoc","co","co2","pm2p5_mass"],study1="UTx000",study2="Ambassador\nFamilies",save=False,**kwargs):
        """
        Plots boxplots to compare distributions of values
        
        Parameters
        ----------
        
        Returns
        -------
        <void>
        """
            
        # getting df in order
        df1["study"] = study1
        df2["study"] = study2
        df = df1.append(df2)
        
        # plotting
        _, axes = plt.subplots(1,len(params),figsize=(14,5),gridspec_kw={"wspace":0.5})
        for param, ax in zip(params,axes):
            df["parameter"] = visualize.get_label(param)
            sns.boxplot(x="parameter",y=param,hue="study",data=df,ax=ax,
                    palette=["seagreen","cornflowerblue"],whis=1.5)
            #x-axis
            ax.set_xlabel("")
            # y-axis
            ax.set_ylim(bottom=0)
            ax.set_ylabel(visualize.get_units(param),fontsize=16)
            # remainder
            ax.tick_params(labelsize=14)
            if param == params[-1]:
                ax.legend(bbox_to_anchor=(-0.25,-0.08),frameon=False,fontsize=14,ncol=2)
            else:
                ax.get_legend().remove()
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
        
        if save:
            s1 = study1.replace('\n','_').lower()
            s2 = study2.replace('\n','_').lower()
            plt.savefig(f"../reports/figures/beacon_summary/boxplot_comparison-{s1}-{s2}.pdf",layout="tight")
        plt.show()
        plt.close()

    def plot_violin_comparison(self,df1,df2,params=["tvoc","co","co2","pm2p5_mass"],study1="UTx000",study2="Ambassador\nFamilies",save=False,**kwargs):
        """
        Plots violins to compare distributions of values
        
        Parameters
        ----------
        
        Returns
        -------
        <void>
        """
        
        # resampling
        if "resample" in kwargs.keys():
            df1 = self.resample_data(df1,kwargs['resample'])
            df2 = self.resample_data(df2,kwargs['resample'])
            
        # getting df in order
        df1["study"] = study1
        df2["study"] = study2
        df = df1.append(df2)
        
        # plotting
        _, axes = plt.subplots(1,len(params),figsize=(14,5),gridspec_kw={"wspace":0.5})
        for param, ax in zip(params,axes):
            df["parameter"] = visualize.get_label(param)
            sns.violinplot(x="parameter",y=param,hue="study",data=df,cut=0,split=True,ax=ax,
                    palette=["seagreen","cornflowerblue"],inner=None)
            #x-axis
            ax.set_xlabel("")
            # y-axis
            ax.set_ylim(bottom=0)
            ax.set_ylabel(visualize.get_units(param),fontsize=16)
            # remainder
            ax.tick_params(labelsize=14)
            if param == params[-1]:
                ax.legend(bbox_to_anchor=(-0.25,-0.08),frameon=False,fontsize=14,ncol=2)
            else:
                ax.get_legend().remove()
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
        
        if save:
            s1 = study1.replace('\n','_').replace(' ','_').lower()
            s2 = study2.replace('\n','_').replace(' ','_').lower()
            plt.savefig(f"../reports/figures/beacon_summary/violin_comparison-{s1}-{s2}.pdf",layout="tight")
        plt.show()
        plt.close()