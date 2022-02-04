from ast import Param
from tracemalloc import start
import pandas as pd
import numpy as np

import math
from datetime import datetime

# user-defined
from src.visualization import visualize

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from joypy import joyplot

class Summarize():

    def __init__(self, study, suffix, data_dir="../data/") -> None:
        """
        Initializing Class

        Parameters
        ----------
        study : str
            specific study name to pull from
        suffix : str

        """
        self.study = study
        self.suffix = suffix
        self.data_dir = data_dir

        self.data = pd.read_csv(f'{self.data_dir}processed/beacon-{self.suffix}.csv',
            index_col=0,parse_dates=True,infer_datetime_format=True)

    def get_beacon(self,beacon_no=1,start_time=None,end_time=None):
        """
        Gets data specific to the given beacon

        Parameters
        ----------
        beacon_no : int, default 1
            number of the beacon
        start_time : datetime, default None
            data start time
        end_time : datetime, default None
            data end time

        Returns
        -------
        data_bb : DataFrame
            data specific to beacon and within time freame (if specified)
        """
        data_bb = self.data[self.data["beacon"] == beacon_no]
        if start_time:
            data_bb = data_bb[start_time:]
        
        if end_time:
            data_bb = data_bb[:end_time]

        return data_bb

    def plot_beacon_ts(self,beacon,param="co2",start_time=None,end_time=None,scatter=False):
        """
        Plots timeseries of the specified parameter within the time range

        Parameters
        ----------
        beacon : int
            number of beacon to pull data from
        params : str, default "co2"
            column to use from data
        start_time : datetime, default None
            data start time
        end_time : datetime, default None
            data end time
        scatter : boolean, default False
            whether to scatter the data versus plot

        Returns
        -------
        <void>
        """
        beacon_df = self.get_beacon(beacon, start_time, end_time)

        _, ax = plt.subplots(figsize=(16,4))
        if scatter:
            ax.scatter(beacon_df.index,beacon_df[param],color="black",s=5,marker="s")
        else:
            ax.plot(beacon_df.index,beacon_df[param],color="black",lw=2)
        # y-axis
        ax.set_ylabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})",fontsize=18)
        # remainder
        ax.tick_params(labelsize=14)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)

    def plot_two_var_ts(self,beacon,params=["co2","pm2p5_mass"],start_time=None,end_time=None,scatter=False,save=False,annot=None):
        """
        Plots timeseries of the specified parameter within the time range

        Parameters
        ----------
        beacon : int
            number of beacon to pull data from
        params : str, default "co2"
            column to use from data
        start_time : datetime, default None
            data start time
        end_time : datetime, default None
            data end time
        scatter : boolean, default False
            whether to scatter the data versus plot

        Returns
        -------
        <void>
        """
        beacon_df = self.get_beacon(beacon, start_time, end_time)

        _, axis = plt.subplots(figsize=(16,4))
        for param, ax, color in zip(params, [axis,axis.twinx()], ["black","cornflowerblue"]):
            if scatter:
                ax.scatter(beacon_df.index,beacon_df[param],color=color,s=5,marker="s")
            else:
                ax.plot(beacon_df.index,beacon_df[param],color=color,lw=2)

            # x-axis
            if start_time:
                ax.set_xlim(left=start_time)
            if end_time:
                ax.set_xlim(right=end_time)
            try:
                if end_time.day - start_time.day <= 14:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            except AttributeError:
                # end_time and start_time not specified
                pass
            
            # y-axis
            ax.set_ylabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})",fontsize=18)
            if param == "co2":
                ax.set_ylim(bottom=400)
            if param == "pm2p5_mass":
                ax.set_ylim(bottom=0)
            if param == params[1]:
                ax.spines['right'].set_color(color)
                ax.tick_params(axis='y', colors=color)
            # remainder
            ax.tick_params(labelsize=14)

        if save:
            if annot:
                plt.savefig(f"/Users/hagenfritz/Desktop/beacon{beacon}-{params[0]}-{params[1]}-{annot}-{self.suffix}.pdf")
            else:
                plt.savefig(f"~/Desktop/beacon{beacon}-{params[0]}-{params[1]}-{self.suffix}.pdf")

        plt.show()
        #plt.close()
    
    def plot_sensor_operation(self,params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]):
        """
        
        """
        for bb in self.data["beacon"].unique():
            bb_df = self.get_beacon(bb)
            bb_df.replace(np.nan,-0.5,inplace=True)
            bb_df[bb_df >= 0] = 0.5
            fig, axes = plt.subplots(len(params),1,figsize=(16,len(params)*1.5),sharex=True,gridspec_kw={"hspace":0.5})
            for param, ax in zip(params, axes):
                on = bb_df[param][bb_df[param] > 0]
                off = bb_df[param][bb_df[param] < 0]
                ax.scatter(on.index,on,color="seagreen",marker="s",s=5)
                ax.scatter(off.index,off,color="firebrick",marker="s",s=5)
                ax.text(1.02,0.5,f"Operation:\n{round(len(on)/len(bb_df)*100,2)}%",ha="left",va="center",transform=ax.transAxes,fontsize=12)
                # x-axis
                ax.set_xlabel("")
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
                ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
                # y-axis
                ax.set_yticks([-0.5,0.5])
                ax.set_yticklabels(["OFF","ON"])
                ax.set_ylim([-1,1])
                # remainder
                ax.tick_params(labelsize=14)
                ax.set_title(visualize.get_label(param),loc="left",fontsize=16)

                fig.add_subplot(111, frame_on=False)
                plt.tick_params(labelcolor="none", bottom=False, left=False)
                plt.ylabel("Sensor Operation",fontsize=18,labelpad=10)
                plt.title(f"Beacon {bb}",fontsize=18)

            plt.show()
            plt.close()

    def get_summary_stats(self, params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"], precision=1,
        save=False, annot=None):
        """
        Gets summary stats for beacon dataset
        
        Parameters
        ----------
        params : list of str, default ["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]
            names of columns to include when summarizing
        precision : int, default 1
            number of decimal points to include
        save : boolean, default False
            whether to save or not
        annot : str, default None
            information to include in the filename when saving

        Returns
        -------
        stats_df : DataFrame
            summary statistics on the variables of the beacon
        """
        df = self.data.copy()
        stats = {'n':[],'avg':[],'med':[],'min':[],'25%':[],'75%':[],'95%':[]}
        for param in params:
            li = df[param].dropna().values
            stats['n'].append(len(li))
            stats['avg'].append(round(np.nanmean(li),precision))
            stats['med'].append(round(np.nanmedian(li),precision))
            stats['min'].append(round(np.nanmin(li),precision))
            stats['25%'].append(round(np.nanpercentile(li,25),precision))
            stats['75%'].append(round(np.nanpercentile(li,75),precision))
            stats['95%'].append(round(np.nanpercentile(li,95),precision))
            
        stats_df = pd.DataFrame(data=stats)
        stats_df.index = params
        print(stats_df.to_latex())
        if save:
            if annot:
                stats_df.to_csv(f'../data/processed/beacon-{annot}-summary_stats-{self.suffix}.csv')
            else:
                stats_df.to_csv(f'../data/processed/beacon-summary_stats-{self.suffix}.csv')
            
        return stats_df

    def plot_correlation_matrix(self, data=None, params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"], save=False, annot="partially_filtered"):
        """
        Plots correlation matrix between variables
        
        Parameters
        ----------
        data : DataFrame, default None
            user-specified data to use, otherwise uses the class data
        params : list of str, default ["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]
            names of columns to include when summarizing
        save : boolean, default False
            whether to save or not
        annot : str, default None
            information to include in the filename when saving

        Creates
        -------
        correlation_matrix : NumPy Array
            correlation matrix between the given parameters
        """
        if isinstance(data,pd.DataFrame):
            df = data[params]
        else:
            df = self.data.copy()[params]
            
        df.columns = [visualize.get_label(col) for col in df.columns]
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

    def plot_distributions(self,data=None,params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]):
        """
        Plots distributions from all the beacons for each given variable

        Parameters
        ----------
        data : DataFrame, default None
            user-specified data to use, otherwise uses the class data
        params : list of str, default ["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]
            names of columns to include when summarizing

        Returns
        -------
        <void>
        """
        if isinstance(data,pd.DataFrame):
            df = data
        else:
            df = self.data.copy()

        for param in params:
            _, ax = plt.subplots(figsize=(12,4))
            sns.kdeplot(param,data=df, ax=ax,
                cut=0, linewidth=2, color="cornflowerblue")
            # x-axis
            ax.set_xlabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})",fontsize=16)
            # y-axis
            ax.set_ylabel("Density",fontsize=16)
            # remainder
            ax.tick_params(labelsize=14)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)


            plt.show()
            plt.close()

    def plot_beacon_joyplots_by_stat(self,data=None,params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"],
        limits=[[400,3000],[0,50],[0,20],[0,1000],[15,30],[20,80]],
        ticks=[range(400,3800,600),range(0,55,5),range(0,24,4),range(0,1200,200),range(15,33,3),range(20,100,20)],
        beacons_to_leave_out = [[],[],[],[],[],[]],
        by_var="beacon",by_stat="mean",save=False,annot=None):
        """
        Plots joyplots for the major sensors on the beacon. 
        
        Parameters
        ----------
        data : DataFrame, default None
            user-specified data to use, otherwise uses the class data
        params : list of str, default ["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]
            names of columns to include when summarizing
        limits : list of lists, default [[400,4000],[0,50],[0,20],[0,1000],[15,30],[20,80]]
            limits to include for the variables
        ticks : list of ranges, default [range(400,3800,600),range(0,55,5),range(0,24,4),range(0,1200,200),range(15,33,3),range(20,100,20)]
        by_var: str, default "beacon"
            column used to separate the individual distributions by
        by_stat: str, default "mean",
            summary stat to order joyplots by
        save : boolean, default False
            whether to save or not
        annot : str, default None
            information to include in the filename when saving
        
        Returns
        -------
        <void>
        """
        if isinstance(data,pd.DataFrame):
            df = data
        else:
            df = self.data.copy()

        titles = ["a","b","c","d","e","f","g","h"]
            
        # looping through the variables 
        for param, limit, tick, title, bad_beacons in zip(params, limits, ticks, titles, beacons_to_leave_out):
            df_filtered = pd.DataFrame()
            for bb in df[by_var].unique():
                if bb in bad_beacons:
                    continue
                else:
                    temp = df[df[by_var] == bb]
                    if by_stat == "median":
                        temp['stat'] = temp[param].median() + 0.0001*int(bb)
                    else:
                        temp['stat'] = temp[param].mean() + 0.0001*int(bb)

                    if math.isnan(temp['stat'][0]):
                        pass
                    else:
                        df_filtered = df_filtered.append(temp)
                
            try:
                df_to_plot = df_filtered[[param,'stat',by_var]].sort_values(["stat"])
                labels = df_to_plot[by_var].unique()
                fig, ax = joyplot(data=df_to_plot,by='stat',column=[param],tails=0,
                    kind='kde',overlap=0.5,ylim="own",range_style="own",x_range=limit,grid="y",
                    labels=labels,alpha=0.75,color="cornflowerblue",figsize=(6,6))
                # x-axis
                plt.xlabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})", fontsize=24)
                plt.xticks(tick,tick)
                # remainder
                fig.suptitle(title, fontsize=28)
                for a in ax:
                    a.tick_params(labelsize=20)
                
                if save:
                    if annot:
                        plt.savefig(f'../reports/figures/beacon_summary/beacon-{param}-joyplot{annot}-{self.suffix}.pdf',bbox_inches="tight")
                    else:
                        plt.savefig(f'../reports/figures/beacon_summary/beacon-{param}-joyplot-{self.suffix}.pdf',bbox_inches="tight")

                plt.show()
                plt.close()
            except AssertionError as e:
                print("Something wrong with labeling")

    def plot_beacon_boxplots_by_stat(self,data=None,params=["co2","pm2p5_mass","co","tvoc","temperature_c","rh"],
        limits=[[400,3000],[0,50],[0,20],[0,1000],[15,30],[20,80]],
        ticks=[range(400,3800,600),range(0,55,5),range(0,24,4),range(0,1200,200),range(15,33,3),range(20,100,20)],
        beacons_to_leave_out = [[],[],[],[],[],[]],
        by_var='beacon',by_stat="mean",save=False,annot=None):
        '''
        Plots joyplots for the major sensors on the beacon. 
        
        Parameters
        ----------
        data : DataFrame, default None
            user-specified data to use, otherwise uses the class data
        params : list of str, default ["co2","pm2p5_mass","co","tvoc","temperature_c","rh"]
            names of columns to include when summarizing
        limits : list of lists, default [[400,4000],[0,50],[0,20],[0,1000],[15,30],[20,80]]
            limits to include for the variables
        ticks : list of ranges, default [range(400,3800,600),range(0,55,5),range(0,24,4),range(0,1200,200),range(15,33,3),range(20,100,20)]
        by_var: str, default "beacon"
            column used to separate the individual distributions by
        by_stat: str, default "mean",
            summary stat to order joyplots by
        save : boolean, default False
            whether to save or not
        annot : str, default None
            information to include in the filename when saving
        
        Returns
        -------
        <void>
        '''

        titles = ["a","b","c","d","e","f"]
        
        if isinstance(data,pd.DataFrame):
            df = data
        else:
            df = self.data.copy()

        for param, limit, tick, title in zip(params, limits, ticks, titles):
            df_filtered = pd.DataFrame()
            for bb in df[by_var].unique():
                temp = df[df[by_var] == bb]
                temp['stat'] = temp[param].median() + 0.0001*int(bb)

                if len(temp) > 0:
                    df_filtered = df_filtered.append(temp)
                    
            _, ax = plt.subplots(figsize=(20,6))
            df_filtered.sort_values("stat",inplace=True,ascending=False)
            sns.boxplot(x="stat",y=param,order=df_filtered["stat"].unique(),data=df_filtered)
            # x-axis
            ax.set_xticklabels(df_filtered["beacon"].unique(),fontsize=14)
            ax.set_xlabel("")
            # y-axis
            plt.yticks(fontsize=14)
            ax.set_ylabel(f"{visualize.get_label(param)} ({visualize.get_units(param)})",fontsize=16)
            ax.set_ylim(limit)
            ax.set_yticks(tick)
            # remainder
            ax.set_title(title,fontsize=20)
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
                
            if annot:
                plt.savefig(f'../reports/figures/beacon_summary/beacon-{param}-boxplot{annot}-{self.suffix}.pdf',bbox_inches="tight")
            else:
                plt.savefig(f'../reports/figures/beacon_summary/beacon-{param}-boxplot-{self.suffix}.pdf',bbox_inches="tight")
            
            plt.show()
            plt.close()

def main():
    pass

if __name__ == '__main__':
    main()