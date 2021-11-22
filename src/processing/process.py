import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import matplotlib.dates as mdates

from scipy import stats
from sklearn.cluster import KMeans

class preprocess():

    def __init__(self, data_dir="../data/",study="wcwh_pilot", suffix="ux_s20", ):
        self.data_dir = data_dir
        self.study = study
        self.suffix = suffix
        self.set_correction_model()

    def set_correction_model(self):
        """
        Sets the class correction models
        """
        # Beacon Attributes
        print("hello")
        self.linear_model = {}
        for file in os.listdir(f"{self.data_dir}/interim/"):
            file_info = file.split("-")
            if len(file_info) == 3:
                if file_info[1] == "linear_model" and file_info[-1] == self.suffix+".csv":
                    try:
                        self.linear_model[file_info[0]] = pd.read_csv(f'{self.data_dir}/interim/{file}',index_col=0)
                    except FileNotFoundError:
                        print(f"Missing linear model for {file_info[0]}")
                        self.linear_model[file_info[0]] = pd.DataFrame(data={"beacon":np.arange(1,51),"constant":np.zeros(51),"coefficient":np.ones(51)}).set_index("beacon")

    def process(self, beacons, start_time=datetime(2020,1,1),end_time=datetime(2022,1,1),save=False):
        """
        generates a processed datafile
        """
        data = pd.DataFrame()
        for beacon in beacons:
            number = f'{beacon:02}'
            data_by_beacon = pd.DataFrame()
            try:
                for file in os.listdir(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/"):
                    if file[-1] == "v":
                        y = int(file.split("_")[1].split("-")[0])
                        m = int(file.split("_")[1].split("-")[1])
                        d = int(file.split("_")[1].split("-")[2].split(".")[0])
                        date = datetime(y,m,d)
                        if date.date() >= start_time.date() and date.date() <= end_time.date():
                            try:
                                temp = pd.read_csv(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/{file}")
                                if len(temp) > 0:
                                    data_by_beacon = data_by_beacon.append(temp)
                            except Exception as e:
                                print("Error with file", file+":", e)
                if len(data_by_beacon) > 0:
                    data_by_beacon["Timestamp"] = pd.to_datetime(data_by_beacon["Timestamp"])
                    data_by_beacon = data_by_beacon.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()[start_time:end_time].resample("1T").mean()
                    data_by_beacon["beacon"] = int(number)
                    data_by_beacon['temperature_c'] = data_by_beacon[['T_CO','T_NO2']].mean(axis=1)
                    data_by_beacon.rename(columns={"Timestamp":"timestamp","TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                    "PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_10":"pm10_number",
                                    "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_10":"pm10_mass"},inplace=True)
                    # correcting
                    print(data_by_beacon.head())
                    for param in ["tvoc","co2","co","temperature_c"]:
                        data_by_beacon[param] = data_by_beacon[param] * self.linear_model[param].loc[beacon,"coefficient"] + self.linear_model[param].loc[beacon,"constant"]

                    data = data.append(data_by_beacon)
            except FileNotFoundError:
                print(f"No files found for beacon {beacon}.")
                
        data['rh'] = data[['RH_CO','RH_NO2']].mean(axis=1)
        data.drop(["eCO2","Visible","Infrared","Relative Humidity","PM_N_0p5","T_CO","T_NO2","RH_CO","RH_NO2"],axis="columns",inplace=True)
        data = data[[column for column in data.columns if "4" not in column]]
        data.reset_index(inplace=True)
        data.rename(columns={"Timestamp":"timestamp","TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                    "PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_10":"pm10_number",
                                    "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_10":"pm10_mass","Temperature [C]":"temperature_c_internal"},inplace=True)
        data["co"] /= 1000
        self.data = data

    def save_data(self, save_dir="../data/processed/",annot=""):
        """
        Saves the data in the specified location
        """
        starting_str = self.data["timestamp"].iloc[0].strftime("%d%m%Y")
        ending_str = self.data["timestamp"].iloc[-1].strftime("%d%m%Y")

        if annot != "":
            annot = "-" + annot

        try:
            self.data.to_csv(f"{save_dir}/beacon-{starting_str}_{ending_str}{annot}.csv")
        except Exception as e:
            print(e)

    def plot_hist(self, df):
        """
        Plots histograms of each of the columns in the given dataframe
        
        Inputs:
        - df: dataframe containing columns with numeric data
        
        Returns void
        """
        _, axes = plt.subplots(1,len(df.columns),figsize=(5*len(df.columns),5),sharey=True)
        colors = cm.get_cmap('Blues_r', len(df.columns))(range(len(df.columns)))
        if len(df.columns) > 1:
            for col, color, ax in zip(df.columns,colors,axes.flat):
                ax.hist(df[col],color=color,rwidth=0.85,edgecolor="black",)
                ax.set_title(col.replace("_"," ").title())
                ax.set_xlim(xmin=0)
                plt.subplots_adjust(wspace=0.1)
                
                for loc in ["right","top"]:
                    ax.spines[loc].set_visible(False)
        else:
            axes.hist(df.iloc[:,0],color=colors,rwidth=0.85,edgecolor="black",)
            axes.set_title(df.columns[0].replace("_"," ").title())
            axes.set_xlim(xmin=0)
            plt.subplots_adjust(wspace=0.1)
            
            for loc in ["right","top"]:
                axes.spines[loc].set_visible(False)
            
        plt.show()
        plt.close()

    def normalize(self, df, method="boxcox"):
        """
        Normalizes each of the columns in the given dataframe according to the method
        
        Inputs:
        - df: dataframe with numeric columns
        - method: string specifying the normalization method
        
        Returns dataframe with normalized columns
        """
        if method == "boxcox":
            for column in df.columns:
                df[column] = stats.boxcox(df[column])[0]
        elif method == "log":
            for column in df.columns:
                df[column] = np.log(df[column])
        elif method == "yeojohnson":
            for column in df.columns:
                df[column] = stats.yeojohnson(df[column])[0]
        return df

    def plot_missing_data(self,df):
        """
        Plots the number of missing values

        Returns void
        """
        missing_vals = df.isnull().sum()
        vals = missing_vals[missing_vals > 0].sort_values()
        # setting up bar chart
        locs = np.arange(len(vals))
        ticks = list(vals.index)
        formatted_ticks = []
        for tick in ticks:
            formatted_ticks.append(tick.replace("_", " ").title())
        my_cmap = plt.get_cmap("Blues")
        rescale = lambda y: y / np.max(y)

        _, ax = plt.subplots(figsize=(8,5))
        rects = ax.barh(locs, vals, color=my_cmap(rescale(vals)), edgecolor="black")
        # formatting y-axis
        plt.yticks(locs, formatted_ticks)
        # formatting remainder
        n = len(df)
        ax.set_title(f"Missing Values - Total Values: {n}")

        for spine_loc in ["top","right"]:
            ax.spines[spine_loc].set_visible(False)
            
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                width = rect.get_width()
                ax.annotate('{}'.format(width),
                            xy=(width, rect.get_y()),
                            xytext=(3, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='left', va='bottom')
                
        autolabel(rects)

        plt.show()
        plt.close()

    def plot_correlation_matrix(self,df,annotate=True):
        """
        Plots correlation matrix between all variables in the df
        
        Inputs:
        - df: dataframe with columns named for the varaiables
        
        """
        corr = df.corr()
        corr = round(corr,2)
        #mask = np.triu(np.ones_like(corr, dtype=bool))
        _, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr,
                        vmin=-1, vmax=1, center=0, 
                        cmap=sns.diverging_palette(20, 220, n=200),cbar_kws={'ticks':[-1,-0.5,0,0.5,1]},
                        square=True,linewidths=0.5,linecolor="black",annot=annotate,fmt=".2f",ax=ax)

        yticklabels = ax.get_yticklabels()
        yticklabels[0] = ' '
        ax.set_yticklabels(yticklabels,rotation=0,ha='right')

        xticklabels = ax.get_xticklabels()
        xticklabels[-1] = ' '
        ax.set_xticklabels(xticklabels,rotation=-45,ha='left')
            
        plt.show()
        plt.close()
