
# General
import os
import math
from datetime import datetime, timedelta
import dateutil.parser
import warnings
from numpy.core.numeric import Inf
# Data Science
import pandas as pd
import numpy as np
from scipy.sparse import base
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import scipy
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.visualization import visualize

class Calibration():

    def __init__(self, start_time, end_time, data_dir="../../data/", study="utx000", study_suffix="ux_s20", **kwargs):
        """
        Initiates the calibration object

        Inputs:
        - start_time: datetime object with precision to the minute specifying the event START time
        - end_time: datetime object with precision to the minute specifying the event END time
        - data_dir: path to data directory
        - study: string of the study name
        - study_suffix: string of the suffix associated with the study

        Keyword Arguments:
        - resample_rate: integer corresponding to the resample rate in minutes - default is 1 minute
        - timestamp: datetime specifying the start time as reported by the laptop
        """
        self.set_start_time(start_time)
        self.set_end_time(end_time)
        if "ref_date" in kwargs.keys():
            self.date = kwargs["ref_date"].date().strftime("%m%d%Y")
        else:
            self.date = end_time.date().strftime("%m%d%Y")

        self.data_dir = data_dir
        self.study = study
        self.suffix = study_suffix
        self.set_time_offset(**kwargs)

        # kwargs
        if "resample_rate" in kwargs.keys():
            self.set_resample_rate(kwargs["resample_rate"])
        else:
            self.set_resample_rate(1) # set to default

        if "beacons" in kwargs.keys():
            self.set_beacons(kwargs["beacons"])
        else:
            self.set_beacons(kwargs["beacons"])

        # data
        ## beacon
        print("IMPORTING BEACON DATA")
        if self.study == "utx000":
            self.set_utx000_beacon(**kwargs)
        else:
            self.set_wcwh_beacon(**kwargs)
        ## refererence
        print("IMPORTING REFERENCE DATA")
        self.ref = {}
        self.set_ref(**kwargs)
        ## calibration
        self.offsets = {}
        self.lms = {}

    # experiment detail setters
    def set_start_time(self, t):
        """sets the calibration start time"""
        self.start_time = t

    def set_end_time(self, t):
        """sets the calibration end_time"""
        self.end_time = t

    def set_resample_rate(self, rate):
        """sets the class resample rate"""
        self.resample_rate = rate

    def set_time_offset(self, **kwargs):
        """
        Sets the offset time for measurements because the laptop time is incorrect
        
        Keyword Arguments:
        - timestamp: datetime specifying the start time as reported by the laptop
        """
        if "version" in kwargs.keys():
            v = kwargs["version"]
        else:
            v = ""
        if "timestamp" in kwargs.keys():
            self.t_offset = self.start_time - kwargs["timestamp"]
        else:
            try:
                # attempting to read pm_mass file to get the starting timestamp recorded by the computer
                temp = pd.read_csv(f"{self.data_dir}calibration/pm_mass_{self.date}{v}.csv",skiprows=6,parse_dates={"timestamp": ["Date","Start Time"]},infer_datetime_format=True)
                self.t_offset = self.start_time - temp["timestamp"].iloc[0]
            except FileNotFoundError:
                print("No file found - try providing a `timestamp` argument instead")
                self.t_offset = 0

    def set_beacons(self, beacon_list):
        """sets the list of beacons to be considered"""
        self.beacons = beacon_list

    # reference setters
    def set_ref(self,ref_species=["pm_number","pm_mass","no2","no","co2","tvoc","co","t","rh"],**kwargs):
        """
        Sets the reference data

        Inputs:
        ref_species: list of strings specifying the reference species data to import
        """
        for species in ref_species:
            if species in ["pm_number", "pm_mass"]:
                self.set_pm_ref(species[3:],**kwargs)
            elif species == "no2":
                self.set_no2_ref(**kwargs)
            elif species == "co2":
                self.set_co2_ref(**kwargs)
            elif species == "no":
                self.set_no_ref(**kwargs)
            elif species == "tvoc" and len(self.beacon_data) > 1:
                self.set_tvoc_ref()
            elif species == "t" or species == "rh":
                self.set_trh_ref(**kwargs)
            else:
                self.set_zero_baseline(species=species)

    def set_zero_baseline(self,species="co"):
        """
        Sets reference of species species to zero (clean) background
        
        Inputs:
        - species: string representing the pollutant species to save to the reference dictionary
        """
        dts = pd.date_range(self.start_time,self.end_time,freq=f'{self.resample_rate}T') # timestamps between start and end
        df = pd.DataFrame(data=np.zeros(len(dts)),index=dts,columns=["concentration"]) # creating dummy dataframe
        df.index.rename("timestamp",inplace=True)
        self.ref[species] = df

    def set_pm_ref(self, concentration_type="mass",**kwargs):
        """
        Sets the reference PM data

        Inputs:
        - concentration_type: string of either "mass" or "number"

        Returns a dataframe with columns PM1, PM2.5, and PM10 indexed by timestamp
        """
        # import data and correct timestamp
        if "version" in kwargs.keys():
            v = kwargs["version"]
        else:
            v = ""
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/pm_{concentration_type}_{self.date}{v}.csv",skiprows=6)
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/pm_{concentration_type}_{self.date}{v}.csv")
            return

        df = raw_data.drop(['Sample #','Aerodynamic Diameter'],axis=1)
        date = df['Date']
        sample_time = df['Start Time']
        datetimes = []
        for i in range(len(date)):
            datetimes.append(datetime.strptime(date[i] + ' ' + sample_time[i],'%m/%d/%y %H:%M:%S') + self.t_offset)

        df['timestamp'] = datetimes
        df.set_index(['timestamp'],inplace=True)
        df = df.iloc[:,:54]
        df.drop(['Date','Start Time'],axis=1,inplace=True)

        # convert all columns to numeric types
        for column in df.columns:
            df[column] = pd.to_numeric(df[column])

        # correct for units
        if concentration_type == "mass":
            factor = 1000
        else:
            factor = 1

        # sum columns for particular size concentrations
        df['pm1'] = df.iloc[:,:10].sum(axis=1)*factor
        df['pm2p5'] = df.iloc[:,:23].sum(axis=1)*factor
        df['pm10'] = df.iloc[:,:42].sum(axis=1)*factor

        # resample
        if "window" in kwargs.keys():
            window = kwargs["window"]
        else:
            window = 5 # defaults to window size of 5
        df_resampled = df.resample(f"{self.resample_rate}T").mean().rolling(window=window,min_periods=1).mean().bfill()
        df_resampled = df_resampled[self.start_time:self.end_time]

        # setting
        for size in ["pm1","pm2p5","pm10"]:
            self.ref[f"{size}_{concentration_type}"] = pd.DataFrame(df_resampled[size]).rename(columns={size:"concentration"})
        
    def set_co2_ref(self,**kwargs):
        """sets the reference CO2 data"""
        if "version" in kwargs.keys():
            v = kwargs["version"]
        else:
            v = ""
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/co2_{self.date}{v}.csv",usecols=[0,1],names=["timestamp","concentration"])
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/co2_{self.date}{v}.csv")
            return 

        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"],yearfirst=True)
        raw_data.set_index("timestamp",inplace=True)
        raw_data.index += self.t_offset# = df.shift(periods=3) 
        if "window" in kwargs.keys():
            window = kwargs["window"]
        else:
            window = 5 # defaults to window size of 5
        df = raw_data.resample(f"{self.resample_rate}T",closed="left").mean().rolling(window=window,min_periods=1).mean().bfill()
        self.ref["co2"] = df[self.start_time:self.end_time]

    def set_trh_ref(self,**kwargs):
        "sets the reference temperature and relative humidity"
        if "version" in kwargs.keys():
            v = kwargs["version"]
        else:
            v = ""
        try:
            raw_data = pd.read_csv(f"../data/calibration/trh_{self.date}{v}.csv",skiprows=11,
                  usecols=["Date","Time","Temp","%RH"],parse_dates=[["Date","Time"]],infer_datetime_format=True)
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/trh_{self.date}{v}.csv")
            return 

        raw_data.columns = ["timestamp","t_c","rh"]
        raw_data.dropna(inplace=True)
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"],yearfirst=False,dayfirst=True)
        raw_data.set_index("timestamp",inplace=True)
        if "window" in kwargs.keys():
            window = kwargs["window"]
        else:
            window = 3 # defaults to window size of 3
        df = raw_data.resample(f"{self.resample_rate}T",closed="left").mean().rolling(window=window,min_periods=1).mean().bfill()
        df = df[self.start_time:self.end_time]
        df_t = pd.DataFrame(df["t_c"])
        df_rh = pd.DataFrame(df["rh"])
        # renamining to match other reference data
        df_t.columns = ["concentration"]
        df_rh.columns = ["concentration"]
        # saving to ref dict
        self.ref["temperature_c"] = df_t
        self.ref["rh"] = df_rh

    def set_no2_ref(self,**kwargs):
        """sets the reference NO2 data"""
        if "version" in kwargs.keys():
            v = kwargs["version"]
        else:
            v = ""
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/no2_{self.date}{v}.csv",usecols=["IgorTime","Concentration"])
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/no2_{self.date}{v}.csv")
            return 

        # Using igor time (time since Jan 1st, 1904) to get timestamp
        ts = []
        for seconds in raw_data["IgorTime"]:
            ts.append(datetime(1904,1,1) + timedelta(seconds=int(seconds))+self.t_offset)

        raw_data["timestamp"] = ts
        raw_data.set_index("timestamp",inplace=True)
        raw_data.drop("IgorTime",axis=1,inplace=True)

        df = raw_data.resample(f"{self.resample_rate}T").mean()
        df.columns = ["concentration"]
        self.ref["no2"] = df[self.start_time:self.end_time]

    def set_no_ref(self,**kwargs):
        """sets the reference no data """
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/no_{self.date}.csv",names=["timestamp","concentration"],skiprows=1)
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/no_{self.date}.csv")
            return
        
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
        raw_data.set_index("timestamp",inplace=True)
        df = raw_data.resample(f"{self.resample_rate}T").mean()
        self.ref["no"] = df[self.start_time:self.end_time]
        
    def set_tvoc_ref(self):
        """sets the tvoc reference as the mean concentration at each timestamp"""
        raw_data = self.beacon_data[["timestamp","tvoc","beacon"]].pivot(index="timestamp",columns="beacon",values="tvoc").dropna(axis=1)
        raw_data["concentration"] = raw_data.mean(axis=1)
        self.ref["tvoc"] = raw_data[["concentration"]]
    # beacon setters
    def set_beacon_data(self,data):
        """sets the beacon data attribute with given data"""
        self.beacon_data = data

    def set_utx000_beacon(self,verbose=False,**kwargs):
        """
        Sets beacon data from utx000 for calibration

        Inputs:
        - beacon_list: list of integers specifying the beacons to consider
        - resample_rate: integer specifying the resample rate in minutes
        - verbose: boolean to have verbose mode on
        """
        self.beacons = kwargs["beacons"]
        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder=f"{self.data_dir}raw/{self.study}/beacon"
        # list of all beacons used in the study
        if verbose:
            print('Processing beacon data...\n\tReading for beacon:')
        for beacon in self.beacons:

            # correcting the number since the values <10 have leading zero in directory
            number = f'{beacon:02}'
            if verbose:
                print(f'\t{number}')

            file_count = 0
            beacon_folder=f'{beacons_folder}/B{number}'
            for file in os.listdir(f'{beacon_folder}/adafruit'):
                if file.endswith('.csv'):
                    file_count += 1
                    
            if file_count > 0:
                beacon_df = pd.DataFrame() # dataframe specific to the beacon

                def import_and_merge(csv_dir,number):
                    df_list = []
                    for file in os.listdir(csv_dir+'/'):
                        try:
                            # reading in raw data (csv for one day at a time) and appending it to the overal dataframe
                            day_df = pd.read_csv(f'{csv_dir}/{file}',
                                                index_col='Timestamp',parse_dates=True,
                                                infer_datetime_format=True)
                            df_list.append(day_df)

                        except Exception:
                            # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                            if verbose:
                                print(f'\t\tIssue encountered while importing {csv_dir}/{file}, skipping...')
                    if "window" in kwargs.keys():
                        window = kwargs["window"]
                    else:
                        window = 5 # defaults to window size of 5
                    df = pd.concat(df_list).resample(f'{self.resample_rate}T').mean().rolling(window=window,min_periods=1).mean().bfill()

                    return df

                # Python3 Sensors
                # ---------------
                py3_df = import_and_merge(f'{beacon_folder}/adafruit', number)

                # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues - see Hagen)
                if int(number) > 27:
                    if verbose:
                        print('\t\tNo NO2 sensor - removing values')

                    py3_df[['CO','T_CO','RH_CO']] = py3_df[['NO2','T_NO2','RH_NO2']]
                    py3_df[['NO2','T_NO2','RH_NO2']] = np.nan

                py3_df['CO'] /= 1000 # converting ppb measurements to ppm

                # Python2 Sensors
                # ---------------
                py2_df = import_and_merge(f'{beacon_folder}/sensirion', number)

                # merging python2 and 3 sensor dataframes
                beacon_df = py3_df.merge(right=py2_df,left_index=True,right_index=True,how='outer')

                # getting relevant data only
                if "start_time" in kwargs.keys():
                    beacon_df = beacon_df[kwargs["start_time"]:]
                else:
                    beacon_df = beacon_df[self.start_time:self.end_time]

                beacon_df.drop(['eCO2','Visible','Infrared',"T_CO","RH_CO","T_NO2","RH_NO2",'Temperature [C]','Relative Humidity','PM_N_0p5','PM_N_4','PM_C_4'],axis=1,inplace=True)

                # concatenating the data to the overall dataframe
                beacon_df['beacon'] = beacon
                beacon_data = pd.concat([beacon_data,beacon_df])
        beacon_data.reset_index(inplace=True)
        beacon_data.columns = ["timestamp","tvoc","light","no2","co","co2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","beacon"]
        
        beacon_data = beacon_data[beacon_data["beacon"] != 0] # correcting for any mislabeled raw data
        self.beacon_data = beacon_data

    def set_wcwh_beacon(self, verbose=False, **kwargs):
        """sets beacon data from wcwh pilot for calibration"""
        data = pd.DataFrame()
        for beacon in self.beacons:
            number = f'{beacon:02}'
            data_by_beacon = pd.DataFrame()
            if verbose:
                print("Beacon", beacon)
            try:
                for file in os.listdir(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/"):
                    if file[-1] == "v":
                        if verbose:
                            print("\t" + file)
                        try:
                            temp = pd.read_csv(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/{file}")
                            if len(temp) > 0:
                                data_by_beacon = data_by_beacon.append(temp)
                        except Exception as e:
                            print("Error with file", file+":", e)
                if len(data_by_beacon) > 0:
                    data_by_beacon["Timestamp"] = pd.to_datetime(data_by_beacon["Timestamp"])
                    data_by_beacon = data_by_beacon.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()[self.start_time:self.end_time].resample(f"{self.resample_rate}T").mean()
                    data_by_beacon["beacon"] = int(number)
                    # looking for any moving mean/median filters
                    if "window" in kwargs.keys():
                        window = kwargs["window"]
                    else:
                        window = 5 # defaults to window size of 5
                    if "moving" in kwargs.keys():
                        if kwargs["moving"] == "median":
                            data = data.append(data_by_beacon.rolling(window=window,min_periods=1).median().bfill())
                        else:
                            data = data.append(data_by_beacon.rolling(window=window,min_periods=1).mean().bfill())
                    else:
                        data = data.append(data_by_beacon)
            except FileNotFoundError:
                print(f"No files found for beacon {beacon}.")
                
        data['temperature_c'] = data[['T_CO','T_NO2']].mean(axis=1)
        data['rh'] = data[['RH_CO','RH_NO2']].mean(axis=1)
        data.drop(["eCO2","Visible","Infrared","Temperature [C]","Relative Humidity","PM_N_0p5","T_CO","T_NO2","RH_CO","RH_NO2"],axis="columns",inplace=True)
        data = data[[column for column in data.columns if "4" not in column]]
        data.reset_index(inplace=True)
        #data.columns = ["timestamp","tvoc","lux","co","no2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","co2","beacon","temperature_c","rh"]
        data.rename(columns={"Timestamp":"timestamp","TVOC":"tvoc","Lux":"lux","NO2":"no2","CO":"co","CO2":"co2",
                                    "PM_N_1":"pm1_number","PM_N_2p5":"pm2p5_number","PM_N_10":"pm10_number",
                                    "PM_C_1":"pm1_mass","PM_C_2p5":"pm2p5_mass","PM_C_10":"pm10_mass"},inplace=True)
        data["co"] /= 1000
        self.beacon_data = data
    
    # beacon getters
    def get_beacon(self,bb):
        """gets beacon data"""
        return self.beacon_data[self.beacon_data["beacon"] == bb]

    # visualizations
    def inspect_by_beacon_by_param(self, species="co2"):
        """5x10 subplot showing timeseries of species"""
        _, axes = plt.subplots(5,10,figsize=(20,10),sharex="col",gridspec_kw={"hspace":0.1,"wspace":0.3})
        for beacon, ax in enumerate(axes.flat):
            data_by_beacon = self.beacon_data[self.beacon_data["beacon"] == beacon].set_index("timestamp")
            ax.plot(data_by_beacon.index,data_by_beacon[species])
            # x-axis
            ax.set_xlim(self.start_time, self.end_time)
            ax.xaxis.set_visible(False)
            # y-axis
            if len(data_by_beacon) == 0:
                ax.yaxis.set_visible(False)
            #remainder
            for loc in ["top","right","bottom"]:
                ax.spines[loc].set_visible(False)
            ax.set_title(beacon,y=1,pad=-6,loc="center",va="bottom")

    def inspect_timeseries(self,species="co2",**kwargs):
        """
        Plots timeseries of all beacons with an operation timeseries given below
        
        Inputs:
        - species: string specifying which ieq parameter to plot

        Keyword Arguments:
        - ylimits: list of two ints or floats specifying the upper and lower bounds
        """
        fig, axes = plt.subplots(2,1,figsize=(20,10),sharex=True,gridspec_kw={"hspace":0})
        for beacon in self.beacon_data["beacon"].unique():
            data_by_beacon = self.beacon_data[self.beacon_data["beacon"] == beacon].set_index("timestamp")
            # timeseries
            ax = axes[0]
            ax.plot(data_by_beacon.index,data_by_beacon[species],marker=visualize.get_marker(int(beacon)),label=beacon)

            ax.set_xlim(left=self.start_time,right=self.end_time)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))

            ax.set_ylabel(visualize.get_pollutant_label(species) + " (" + visualize.get_pollutant_units(species) +")",fontsize=14)
            if "ylimits" in kwargs.keys():
                ax.set_ylim(kwargs["ylimits"])

            ax.legend(title="Beacon",ncol=2,bbox_to_anchor=(1,1),frameon=False,title_fontsize=12,fontsize=10)
            # operation
            ax = axes[1]
            data_by_beacon["op"] = data_by_beacon[species].notna()
            ax.scatter(data_by_beacon.index,data_by_beacon["op"]+int(beacon)/50,marker=visualize.get_marker(int(beacon)),s=10,label=beacon)
            # x-axis
            ax.set_xlim(left=self.start_time,right=self.end_time)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            # y-axis
            ax.set_ylim([-0.1,2.1])
            # legend
            ax.legend(title="Beacon",ncol=2,bbox_to_anchor=(1,1),frameon=False,title_fontsize=12,fontsize=10)
        
        plt.show()
        plt.close()
            
    def inspect(self,df,timeseries=True):
        """
        Visually inspect data in dataframe

        Inputs:
        - df: dataframe with one column with values or column named "beacons" that includes the beacon number
        - timeseries: boolean specifying whether or not to plot the timeseries or not (therefore heatmap)
        """

        if timeseries:
            _, ax = plt.subplots(figsize=(16,6))
            if "beacon" in df.columns:
                for bb in df["beacon"].unique():
                    df_by_bb = df[df["beacon"] == bb]
                    ax.plot(df_by_bb.index,df_by_bb.iloc[:,0].values,marker=self.get_marker(int(bb)),label=bb)
            else:
                ax.plot(df.index,df.iloc[:,0].values,linewidth=3,color="black",label="Ref")

            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()
        else: #heatmap
            _, ax = plt.subplots(figsize=(14,7))
            if "beacon" in df.columns:
                df.columns=["concentration","beacon"]
                df_to_plot = pd.DataFrame()
                for bb in df["beacon"].unique():
                    df_by_bb = df[df["beacon"] == bb]
                    df_by_bb.drop("beacon",axis=1,inplace=True)
                    df_to_plot = pd.concat([df_to_plot,df_by_bb],axis=1)
                    df_to_plot.rename(columns={"concentration":bb}, inplace=True)

                sns.heatmap(df_to_plot.T,vmin=np.nanmin(df_to_plot),vmax=np.nanmax(df_to_plot),ax=ax)
                locs, labels = plt.xticks()
                new_labels = []
                for label in labels:
                    new_labels.append(dateutil.parser.isoparse(label.get_text()).strftime("%m-%d-%y %H:%M"))
                plt.xticks(locs,new_labels,rotation=-45,ha="left")
                plt.yticks(rotation=0,va="center")
            else:
                sns.heatmap(df.T,vmin=np.nanmin(df),vmax=np.nanmax(df),ax=ax)

    # visuals
    def compare_time_series(self,species,**kwargs):
        """
        Plots reference and beacon data as a time series

        Inputs:
        - species: string specifying which ieq parameter to plot

        Keyword Arguments:
        - ax: 
        - beacons: 
        - data:
        """
        if "ax" in kwargs.keys():
            ax = kwargs["ax"]
        else:
            _, ax = plt.subplots(figsize=(17,6))
        # plotting reference
        ax.plot(self.ref[species].index,self.ref[species].iloc[:,0].values,linewidth=3,color="black",zorder=100,label="Reference")
        # plotting beacon data
        if "beacons" in kwargs.keys():
            beacon_list = kwargs["beacons"]
        else:
            beacon_list = self.beacon_data["beacon"].unique()
        for bb in beacon_list:    
            if "data" in kwargs.keys():
                data_by_bb = kwargs["data"]
            else:
                data_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
                
            data_by_bb.dropna(subset=[species],inplace=True)
            if len(data_by_bb) > 0:
                ax.plot(data_by_bb.index,data_by_bb[species],marker=visualize.get_marker(int(bb)),zorder=int(bb),label=bb)
            
        # x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(10,60,10)))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
        ax.set_xlim([self.start_time,self.end_time])
        # y_axis
        ax.set_ylabel("Concentration",fontsize=14)
        # remainder
        ax.tick_params(axis="both",labelsize=12)
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05),frameon=False,title="Device",title_fontsize=12,fontsize=10,ncol=25)
            
        if "ax" in kwargs.keys():
            return ax

        plt.show()
        plt.close()

    def compare_scatter(self,species,**kwargs):
        """
        Scatter of measured points between beacons and reference
        
        Inputs:

        Keyword Arguments:
        - ax: 
        - beacons: 
        - data:
        - min_val:
        """
        if "beacons" in kwargs.keys():
            beacon_list = kwargs["beacons"]
        else:
            beacon_list = self.beacon_data["beacon"].unique()
        
        for bb in beacon_list: 
            # getting data to plot   
            if "data" in kwargs.keys():
                data_by_bb = kwargs["data"]
            else:
                data_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
        
            comb = data_by_bb.merge(right=self.ref[species],left_index=True,right_index=True)

            if "ax" in kwargs.keys():
                ax = kwargs["ax"]
            else:
                _, ax = plt.subplots(figsize=(8,8))

            im = ax.scatter(comb["concentration"],comb[species],c=comb.index,cmap="Blues",edgecolor="black",s=50,label="Measured",zorder=2)
            #fig.colorbar(im,ax=ax,label="Minutes since Start")
            max_val = max(np.nanmax(comb["concentration"]),np.nanmax(comb[species]))
            if "min_val" in kwargs.keys():
                min_val = kwargs["min_val"]
            else:
                min_val = 0
            # 1:1
            ax.plot([min_val,max_val],[min_val,max_val],color="firebrick",linewidth=2,zorder=1)
            # x-axis
            ax.set_xlabel("Reference Measurement",fontsize=14)
            ax.set_xlim(left=min_val,right=max_val)
            # y-axis
            ax.set_ylabel("Beacon Measurement",fontsize=14)
            ax.set_ylim(bottom=min_val,top=max_val)
            # remainder
            ax.tick_params(axis="both",labelsize=12)
            ax.set_title(bb)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if "ax" in kwargs.keys():
                return ax

            plt.show()
            plt.close()

    def show_linear_correction(self,species,**kwargs):
        """plots the original and corrected data against the reference for the given species"""
        for bb in self.beacon_data["beacon"].unique():
            beacon_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
            #beacon_by_bb = self.apply_laplacion_filter(beacon_by_bb,species)
            try:
                _, axes = plt.subplots(1,4,figsize=(26,6),gridspec_kw={"wspace":0.2,"width_ratios":[0.25,0.25,0.25,0.25]})
                self.compare_time_series(species,ax=axes[0],beacons=[bb])
                # original data - scatter
                self.compare_scatter(species,ax=axes[1],beacons=[bb],**kwargs)
                # corrected data - timeseries
                corrected_by_bb = beacon_by_bb.copy()
                corrected_by_bb[species] = beacon_by_bb[species] * self.lms[species].loc[bb,"coefficient"] + self.lms[species].loc[bb,"constant"]
                corrected_by_bb = corrected_by_bb.shift(self.lms[species].loc[bb,"ts_shift"])[:len(self.ref[species])]
                self.compare_time_series(species,ax=axes[2],beacons=[bb],data=corrected_by_bb)
                # corrected data - scatter
                self.compare_scatter(species,ax=axes[3],beacons=[bb],data=corrected_by_bb,**kwargs)
                plt.show()
                plt.close()
            except ValueError as e:
                print(e)
                print(f"Length of data for Beacon {bb} is {len(beacon_by_bb[species].dropna())}")
            except KeyError as e:
                print(e)
                print("No data for beacon", bb)

    def show_comprehensive_calibration(self,**kwargs):
        """shows the three figure panel of the calibration"""
        for bb in self.beacon_data["beacon"].unique():
            beacon_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
            if "start_time" in kwargs.keys():
                beacon_by_bb = beacon_by_bb[kwargs["start_time"]:]
            if "end_time" in kwargs.keys():
                beacon_by_bb = beacon_by_bb[:kwargs["end_time"]]

            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            # top timeseries figure
            ts = fig.add_subplot(gs[0,:])
            # bottom left correlation plot
            corr = fig.add_subplot(gs[1,0])
            # bottom right difference plot
            diff = fig.add_subplot(gs[1,1])

    def show_comprehensive_linear_corr(self,species,r,c,**kwargs):
        """shows a subplot of all the correlation beacons"""
        fig, axes = plt.subplots(r,c,figsize=(c*4,r*4),sharex=True,sharey=True)
        for bb, ax in zip(self.beacons,axes.flat):
            beacon_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
            corrected_by_bb = beacon_by_bb.copy()
            corrected_by_bb[species] = beacon_by_bb[species] * self.lms[species].loc[bb,"coefficient"] + self.lms[species].loc[bb,"constant"]
            corrected_by_bb = corrected_by_bb.shift(self.lms[species].loc[bb,"ts_shift"])[:len(self.ref[species])]
            ax.scatter(self.ref[species]["concentration"],corrected_by_bb[species],color="black",zorder=2)
            max_val = max(np.nanmax(self.ref[species]["concentration"]),np.nanmax(corrected_by_bb[species]))
            if "min_val" in kwargs.keys():
                min_val = kwargs["min_val"]
            else:
                min_val = 0
            # 1:1
            ax.plot([min_val,max_val],[min_val,max_val],color="firebrick",linewidth=2,zorder=1)
            # axis

            # annotating
            lm_bb = self.lms[species][self.lms[species].index == bb]
            r2 = self.lms[species]
            ax.set_title(f"  Device {bb}\n  r$^2$ = {round(lm_bb['score'].values[0],3)}\n  y = {round(lm_bb['coefficient'].values[0],1)}x + {round(lm_bb['constant'].values[0],1)}",
                        y=0.85,pad=0,fontsize=13,loc="left",ha="left")
            ax.axis('off')

        axes[r-1,0].axis('on')
        for loc in ["top","right"]:
            axes[r-1,0].spines[loc].set_visible(False)
        plt.setp(axes[r-1,0].get_xticklabels(), ha="center", rotation=0, fontsize=16)
        plt.setp(axes[r-1,0].get_yticklabels(), ha="right", rotation=0, fontsize=16)
        axes[1,0].text(-1,7.5,f'BEVO Beacon {visualize.get_pollutant_label(species)} ({visualize.get_pollutant_units(species)})',rotation=90,ha='center',va='center',fontsize=18)
        axes[r-1,3].text(7.5,1,f'Reference {visualize.get_pollutant_label(species)} ({visualize.get_pollutant_units(species)})',ha='center',va='top',fontsize=18)

        plt.show()
        plt.close()

    def show_step_offset(self,species="co",base_vals=[0,1,2,4],step_length=2):
        """
        Visualizes results from step calibration offset

        Parameters
        ----------
        species : str
            Variable of interest
        base_vals : list of int/float, default [0,1,2,4]
            List of base values at each step
        step_length : int or float, default 2
            Length of each step in the experiment in hours

        Returns
        -------
        <void>
        """
        ref_index = pd.date_range(start=self.beacon_data["timestamp"].iloc[0],
                                end=self.beacon_data["timestamp"].iloc[0]+timedelta(hours=step_length*len(base_vals)),
                                freq=f"{self.resample_rate}T",closed="right")
        ref_vals = []
        for val in base_vals:
            ref_vals += [val]*int(step_length*60/self.resample_rate)

        _, ax = plt.subplots(figsize=(16,4))
        ax.plot(ref_index,ref_vals,lw=2,color="black",label="Reference")
        for bb in self.beacon_data["beacon"].unique():
            data_bb = self.beacon_data[self.beacon_data["beacon"] == bb]
            offset_val = self.offsets[species].loc[bb,"constant"]
            ax.plot(data_bb["timestamp"],data_bb[species]-offset_val,lw=1,marker=visualize.get_marker(int(bb)),zorder=int(bb),label=bb)
        
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend()

        plt.show()
        plt.close()
        #return pd.DataFrame(data=ref_vals,index=ref_index,columns=["concentration"])

    def show_step_linear(self,species="co",base_vals=[0,1,2,4],step_length=2):
        """
        Shows the results from the linear correction on the step calibration
        """
        ref_index = pd.date_range(start=self.beacon_data["timestamp"].iloc[0],
                                end=self.beacon_data["timestamp"].iloc[0]+timedelta(hours=step_length*len(base_vals)),
                                freq=f"{self.resample_rate}T",closed="right")
        ref_vals = []
        for val in base_vals:
            ref_vals += [val]*int(step_length*60/self.resample_rate)

        _, ax = plt.subplots(figsize=(16,4))
        ax.plot(ref_index,ref_vals,lw=2,color="black",label="Reference")
        for bb in self.beacon_data["beacon"].unique():
            data_bb = self.beacon_data[self.beacon_data["beacon"] == bb]
            y = data_bb[species] * self.lms[species].loc[bb,"coefficient"] + self.lms[species].loc[bb,"constant"]
            ax.plot(data_bb["timestamp"],y,lw=1,marker=visualize.get_marker(int(bb)),zorder=int(bb),label=bb)
        
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend()

        plt.show()
        plt.close()

    def show_comprehensive_ts(self,species,r,c,beacons_to_exclude=[],save=False,**kwargs):
        """Plots comprehensive time series of the species against the min and max values"""
        data = self.beacon_data[~self.beacon_data["beacon"].isin(beacons_to_exclude)]
        temp = data[["timestamp",species,"beacon"]].pivot(index="timestamp",columns="beacon",values=species)#.dropna(axis=1)
        for col in temp.columns:
            offset = self.offsets[species][self.offsets[species].index == col]
            temp[col] += offset["constant"].values

        temp["mean"] = temp.mean(axis=1)
        temp["min"] = temp.min(axis=1)
        temp["max"] = temp.max(axis=1)
        temp["t"] = (temp.index - temp.index[0]).total_seconds()/60
        fig, axes = plt.subplots(r,c,figsize=(c*4,r*4),sharex=True,sharey=True)
        for bb, ax in zip(self.beacons,axes.flat):
            try:
                ax.plot(temp["t"],temp[bb],color="black",linewidth=2,zorder=2)
                ax.fill_between(temp["t"],temp["min"],temp["max"],alpha=0.5,color="grey",zorder=1)
            except KeyError:
                pass
            ax.set_title(f"  Device {int(bb)}",y=0.85,pad=0,fontsize=13,loc="left",ha="left")
            ax.axis("off")
            if "limits" in kwargs.keys():
                #ax.set_xlim(kwargs["limits"])
                ax.set_ylim(kwargs["limits"])
        
        axes[r-1,0].axis('on')
        for loc in ["top","right"]:
            axes[r-1,0].spines[loc].set_visible(False)
        ax.set_xticks(np.arange(0,125,30))
        plt.setp(axes[r-1,0].get_xticklabels(), ha="center", rotation=0, fontsize=16)
        plt.setp(axes[r-1,0].get_yticklabels(), ha="right", rotation=0, fontsize=16)
        axes[1,0].text(-2,7.5,f"Concentration ({visualize.get_pollutant_units(species)})",rotation=90,ha='right',va='center',fontsize=24)
        axes[r-1,int(c/2)].text(7.5,-2,f'Experiment Time (minutes)',ha='center',va='top',fontsize=24)
        if save:
            if "study" in kwargs.keys():
                study = "-"+kwargs["study"]
            else:
                study = ""
            plt.savefig(f"../reports/figures/beacon_summary/calibration-{species}-comprehensive_ts{study}.pdf",bbox_inches="tight")
        plt.show()
        plt.close()
    # deprecated
    def compare_histogram(self,ref_data,beacon_data,bins):
        """
        Plots reference and beacon data as histograms

        Inputs:
        - ref_data: dataframe of reference data with single column corresponding to data indexed by time
        - beacon_data: dataframe of beacon data with two columns corresponding to data and beacon number indexed by time
        """
        _, ax = plt.subplots(10,5,figsize=(30,15),gridspec_kw={"hspace":0.5})  
        for i, axes in enumerate(ax.flat):
            # getting relevant data
            beacon_df = beacon_data[beacon_data["beacon"] == i]
            beacon_df.dropna(inplace=True)
            if len(beacon_df) > 1:
                # reference data
                axes.hist(ref_data.iloc[:,0].values,bins=bins,color="black",zorder=1)
                # beacon data
                axes.hist(beacon_df.iloc[:,0].values,bins=bins,color="white",edgecolor="black",alpha=0.7,zorder=9) 
                axes.set_title(f"B{i}",pad=0)   
                # x-axis
                axes.set_xticks(bins)
                # remainder
                for spine in ["top","right"]:
                    axes.spines[spine].set_visible(False)   
            else:
                # making it easier to read by removing the unused figures
                axes.set_xticks([])
                axes.set_yticks([])
                for spine in ["top","right","bottom","left"]:
                    axes.spines[spine].set_visible(False)  
            
        plt.show()
        plt.close()

    # diagnostics
    def get_reporting_beacons(self,species):
        """gets the list of beacons that report measurements from the specified sensor"""
        var_only = self.beacon_data[[species,"beacon"]]
        reporting_beacons = []
        for bb in var_only["beacon"].unique():
            df = var_only.dropna(subset=[species])
            if len(df) > 2:
                reporting_beacons.append(bb)
        try:
            if species.lower() == "no2":
                possible_beacons = [x for x in self.beacons if x <= 28] # getting no2 sensing beacons only
                missing_beacons = [x for x in possible_beacons if x not in reporting_beacons]
            else:
                missing_beacons = [x for x in self.beacons if x not in reporting_beacons]

            print(f"Missing data from: {missing_beacons}")
        except AttributeError:
            print("Calibration object has no attribute 'beacons' - run 'set_beacons' with the desired beacon list")
            return [], reporting_beacons

        return missing_beacons, reporting_beacons

    # calibration
    def offset(self,species,baseline=0,save_to_file=False,show_corrected=False):
        """
        Gets the average offset value and standard deviation between the beacon and reference measurement

        Inputs:
        - species: string specifying the variable of interest

        Returns dataframe holding the average difference and standard deviation between the differences
        """
        offsets = {"beacon":[],"mean_difference":[],"value_to_baseline":[],"constant":[]}
        ref_df = self.ref[species]

        for beacon in np.arange(1,51):
            offsets["beacon"].append(beacon)
            # getting relevant data
            beacon_df = self.beacon_data[self.beacon_data["beacon"] == beacon].set_index("timestamp")
            beacon_df.dropna(subset=[species],inplace=True)
            if len(beacon_df) > 1:
                if len(ref_df) != len(beacon_df):
                    # resizing arrays to include data from both modalities
                    max_start_date = max(ref_df.index[0],beacon_df.index[0])
                    min_end_date = min(ref_df.index[-1],beacon_df.index[-1])
                    ref_df = ref_df[max_start_date:min_end_date]
                    beacon_df = beacon_df[max_start_date:min_end_date]
                    print(f"Beacon {beacon}: Reference and beacon data are not the same length")
                # merging beacon and reference data to get difference
                for shift in range(4):
                    temp_beacon = beacon_df.copy()
                    temp_beacon.index += timedelta(minutes=shift)
                    df = pd.merge(left=temp_beacon,left_index=True,right=ref_df,right_index=True,how="inner")
                    if len(df) > 1:
                        beacon_df.index += timedelta(minutes=shift)
                        break

                df["delta"] = df[species] - df["concentration"]
                # adding data
                mean_delta = np.nanmean(df["delta"])
                val_to_base = np.nanmin(df[species]) - baseline
                offsets["mean_difference"].append(mean_delta)
                offsets["value_to_baseline"].append(val_to_base)
                if np.nanmin(df[species]) - mean_delta < baseline:
                    offsets["constant"].append((np.nanmin(df[species]) - baseline)*-1)
                else:
                    offsets["constant"].append(mean_delta*1)     
            else:
                # adding zeros
                offsets["mean_difference"].append(0)
                offsets["value_to_baseline"].append(0)
                offsets["constant"].append(0)

        offset_df = pd.DataFrame(data=offsets)
        offset_df.set_index("beacon",inplace=True)
        self.offsets[species] = offset_df
        if save_to_file:
            self.save_offsets(species)

        if show_corrected:
            # Plotting Corrected Timeseries by beacon
            # ---------------------------------------
            fig, ax = plt.subplots(10,5,figsize=(30,15),sharex=True)  
            for i, axes in enumerate(ax.flat):
                # getting relevant data
                beacon_df = self.beacon_data[self.beacon_data["beacon"] == i]
                beacon_df.dropna(subset=[species],inplace=True)
                if len(beacon_df) > 1:
                    axes.plot(ref_df.index,ref_df["concentration"],color="black")

                    beacon_df[species] -= offset_df.loc[i,"constant"]
                    axes.plot(beacon_df.index,beacon_df[species],color="seagreen")
                    axes.set_title(f"beacon {i}")  
                    for spine in ["top","right","bottom"]:
                        axes.spines[spine].set_visible(False)         
                else:
                    # making it easier to read by removing the unused figures
                    axes.set_xticks([])
                    axes.set_yticks([])
                    for spine in ["top","right","bottom","left"]:
                        axes.spines[spine].set_visible(False)    

            plt.subplots_adjust(hspace=0.5)
            plt.show()
            plt.close()

            # Plotting Corrected Timeseries over Entire Calibration
            # -----------------------------------------------------
            fig, ax = plt.subplots(figsize=(16,6))
            for bb in self.beacon_data["beacon"].unique():
                beacon_df = self.beacon_data[self.beacon_data["beacon"] == bb]
                beacon_df.dropna(subset=[species],inplace=True)
                if len(beacon_df) > 1:
                    beacon_df[species] -= offset_df.loc[bb,"constant"]
                    ax.plot(beacon_df.index,beacon_df[species],marker=self.get_marker(int(bb)),zorder=int(bb),label=bb)
            
            for spine in ["top","right"]:
                ax.spines[spine].set_visible(False) 

            ax.plot(ref_df.index,ref_df["concentration"],color="black",zorder=99)
            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()

    def step_calibration_offset(self,species="co",base_vals=[0,1,2,4],step_length=2,trim=0.25):
        """
        Gets offset values based on step calibration

        Parameters
        ----------
        species : str
            Variable of interest
        base_vals : list of int/float, default [0,1,2,4]
            List of base values at each step
        step_length : int or float, default 2
            Length of each step in the experiment in hours
        trim : float, default 0.25
            Fraction of step_length to trim from beginning and end

        Returns
        -------
        offsets : dict
            List of offsets corresponding to each reference base level
        """
        offsets = {base: [] for base in base_vals}
        offsets["beacon"] = []
        for bb in self.beacon_data["beacon"].unique():
            offsets["beacon"].append(bb)
            data_bb = self.beacon_data[self.beacon_data["beacon"] == bb]
            data_bb.set_index("timestamp",inplace=True)
            start_time = data_bb.index[0]
            for step in range(len(base_vals)):
                step_start = start_time+timedelta(hours=step_length*(step))+timedelta(hours=step_length*trim)
                step_end = start_time+timedelta(hours=step_length*(step+1))-timedelta(hours=step_length*trim)
                data_bb_step = data_bb[step_start:step_end]
                offsets[base_vals[step]].append(np.nanmean(data_bb_step[species]) - base_vals[step])

        exp_offsets = pd.DataFrame(offsets)
        exp_offsets.set_index("beacon",inplace=True)
        temp_offsets = {"beacon":[],"mean_difference":[],"value_to_baseline":[],"constant":[]}
        for key,val in zip(temp_offsets.keys(),[offsets["beacon"],exp_offsets.mean(axis=1).values,exp_offsets.mean(axis=1).values,exp_offsets.mean(axis=1).values]):
            temp_offsets[key] = val

        self.offsets[species] = pd.DataFrame(temp_offsets).set_index("beacon")
        return exp_offsets

    def step_calibration_linear(self,species="co",base_vals=[0,1,2,4],step_length=2,trim=0.25):
        """
        Gets linear fit from step calibration

        Parameters
        ----------
        species : str
            Variable of interest
        base_vals : list of int/float, default [0,1,2,4]
            List of base values at each step
        step_length : int or float, default 2
            Length of each step in the experiment in hours
        trim : float, default 0.25
            Fraction of step_length to trim from beginning and end

        Returns
        -------
        params : dict
            List of offsets corresponding to each reference base level
        """
        n = len(self.beacon_data["beacon"].unique())
        _, axes = plt.subplots(1,n,figsize=(4*n,4))
        coeffs = {"beacon":[],"constant":[],"coefficient":[],"score":[],"ts_shift":[]}
        for bb,ax in zip(self.beacon_data["beacon"].unique(),axes.flat):
            coeffs["beacon"].append(bb)
            data_bb = self.beacon_data[self.beacon_data["beacon"] == bb]
            data_bb.set_index("timestamp",inplace=True)
            start_time = data_bb.index[0]
            x = []
            for step in range(len(base_vals)):
                step_start = start_time+timedelta(hours=step_length*(step))+timedelta(hours=step_length*trim)
                step_end = start_time+timedelta(hours=step_length*(step+1))-timedelta(hours=step_length*trim)
                data_bb_step = data_bb[step_start:step_end]
                x.append(np.nanmean(data_bb_step[species]))

            x = np.array(x)
            regr = linear_model.LinearRegression()
            regr.fit(x.reshape(-1, 1), base_vals)
            for param, label in zip([regr.intercept_, regr.coef_[0], regr.score(x.reshape(-1, 1),base_vals),0], ["constant","coefficient","score","ts_shift"]):
                coeffs[label].append(param)
            
            ax.scatter(base_vals,x,color="black",s=10)
            x_vals = np.linspace(0,max(base_vals),100)
            ax.plot(base_vals,regr.intercept_+x*regr.coef_[0],color="firebrick",lw=2)

        plt.show()
        plt.close()
        coeff_df = pd.DataFrame(coeffs)
        coeff_df.set_index("beacon",inplace=True)
        self.lms[species] = coeff_df

        return coeffs

    def get_linear_model_params(self,df,x_label,y_label,**kwargs):
        """runs linear regression and returns intercept, slope, r2, and mae"""
        x = df.loc[:,x_label].values
        y = df.loc[:,y_label].values

        regr = linear_model.LinearRegression()
        try:
            if "weights" in kwargs.keys():
                weights = kwargs["weights"]
            else:
                weights= None
            regr.fit(x.reshape(-1, 1), y, sample_weight=weights)
            y_pred = regr.intercept_ + x * regr.coef_[0]
            return regr.intercept_, regr.coef_[0], regr.score(x.reshape(-1, 1),y), mean_absolute_error(y_true=y,y_pred=y_pred)
        except ValueError as e:
            print(f"Error with data ({e}) - returning (0,1)")
            return 0, 1, 0, np.nan

    def apply_laplacion_filter(self,data,var,threshold=0.25):
        """applies laplacian filter to data and returns values with threshold limits"""
        lap = scipy.ndimage.filters.laplace(data[var])
        lap /= np.max(lap)
        # filtering out high variability
        data["lap"] = lap
        data_filtered = data[(data["lap"] < threshold) & (data["lap"] > -1*threshold)]
        data_filtered.drop("lap",axis="columns",inplace=True)
        return data_filtered
    
    def linear_regression(self,species,weight=False,save_to_file=False,verbose=False,**kwargs): 
        """generates a linear regression model"""
        coeffs = {"beacon":[],"constant":[],"coefficient":[],"score":[],"mae":[],"ts_shift":[]}
        ref_df = self.ref[species]
        data = self.beacon_data[["timestamp",species,"beacon"]]
        for bb in np.arange(1,51):
            beacon_by_bb = data[data["beacon"] == bb].set_index("timestamp")
            if verbose:
                print(f"Working for Beacon {bb}")
                print(beacon_by_bb.head())
            if len(beacon_by_bb) > 1:
                if len(ref_df) != len(beacon_by_bb):
                    # resizing arrays to include data from both modalities
                    max_start_date = max(ref_df.index[0],beacon_by_bb.index[0])
                    min_end_date = min(ref_df.index[-1],beacon_by_bb.index[-1])
                    ref_df = ref_df[max_start_date:min_end_date]
                    beacon_by_bb = beacon_by_bb[max_start_date:min_end_date]
                    print(f"Beacon {bb}: Reference and beacon data are not the same length")
                
                beacon_by_bb.drop(["beacon"],axis=1,inplace=True)
                # applying laplacion filter
                if "lap_filter" in kwargs.keys():
                    if kwargs["lap_filter"] == True:
                        beacon_by_bb = self.apply_laplacion_filter(beacon_by_bb,species)

                # running linear models with shifted timestamps
                best_params = [-math.inf,-math.inf,-math.inf, -math.inf] # b, m, r2, ts_shift
                for ts_shift in range(-3,4):
                    comb = ref_df.merge(right=beacon_by_bb.shift(ts_shift),left_index=True,right_index=True,how="inner")
                    comb.dropna(inplace=True)
                    if "event" in kwargs.keys():
                        event = kwargs["event"]
                        data_before_event = comb[:event]
                        baseline = np.nanmean(data_before_event[species])
                        data_before_event = data_before_event[data_before_event[species] > baseline-np.nanstd(data_before_event[species])]
                        data_after_event = comb[event:]
                        data_after_event = data_after_event[data_after_event[species] > baseline+2*np.nanstd(data_before_event[species])]
                        comb = pd.concat([data_before_event,data_after_event])

                    if weight:
                        b, m, r2, mae = self.get_linear_model_params(comb,species,"concentration",weights=comb["concentration"])
                    else:
                        b, m, r2, mae = self.get_linear_model_params(comb,species,"concentration")

                    if r2 > best_params[2]:
                        best_params = [b, m, r2, mae, ts_shift]

                # adding data
                coeffs["beacon"].append(bb)
                for param, label in zip(best_params, ["constant","coefficient","score","mae","ts_shift"]):
                    coeffs[label].append(param)

            else:
                # adding base values
                coeffs["beacon"].append(bb)
                for param, label in zip([0,1,0,np.nan,0], ["constant","coefficient","score","mae","ts_shift"]):
                    coeffs[label].append(param)

        coeff_df = pd.DataFrame(coeffs)
        coeff_df.set_index("beacon",inplace=True)
        self.lms[species] = coeff_df
        if save_to_file:
            self.save_lms(species)

    # saving
    def save_offsets(self,species,**kwargs):
        """saves offset results to file"""
        if "version" in kwargs.keys():
            v = "_" + kwargs["version"]
        else:
            v = ""
        try:
            self.offsets[species].to_csv(f"{self.data_dir}interim/{species.lower()}-constant_model{v}-{self.suffix}.csv")
        except KeyError:
            print("Offset has not been generated for species", species)

    def save_lms(self, species, **kwargs):
        """saves linear model results to file"""
        if "version" in kwargs.keys():
            v = "_" + kwargs["version"]
        else:
            v = ""
        try:
            self.lms[species].to_csv(f"{self.data_dir}interim/{species.lower()}-linear_model{v}-{self.suffix}.csv")
        except KeyError:
            print("Linear model has not been generated for species", species)

    def save_reference(self, species, env):
        """
        Saves the reference data

        Parameters
        ----------
        species : str
            parameter reference data to save
        env : str
            calibration environment_label
        """
        try:
            self.ref[species].to_csv(f"{self.data_dir}interim/{species.lower()}-reference-{env}-{self.suffix}.csv")
        except KeyError:
            print("No reference data for", species)

    def save_test_data(self,species, env):
        """
        Saves the test data for the given parameter

        Parameters
        ----------
        species : str
            parameter reference data to save
        env : str
            calibration environment_label
        """
        try:
            self.beacon_data[["timestamp","beacon",species]].to_csv(f"{self.data_dir}interim/{species.lower()}-test_data-{env}-{self.suffix}.csv")
        except KeyError:
            print("No reference data for", species)

class Model_Comparison():

    def __init__(self,model1_coeffs, model2_coeffs,label1="M1",label2="M2",model_type="linear",**kwargs):
        self.model1_coeffs = model1_coeffs
        self.label1 = label1
        self.model2_coeffs = model2_coeffs
        self.label2 = label2

    def compare_coeffs(self,species="co2",save=False):
        """Compares the coefficients from the two models"""
        combined = self.model1_coeffs.merge(right=self.model2_coeffs,left_index=True,right_index=True,suffixes=("_1","_2"))
        combined.reset_index(inplace=True)
        for coeff in ["constant","coefficient"]:
            # getting new metrics
            combined[f"{coeff}_mean"] = combined[[f"{coeff}_1",f"{coeff}_2"]].mean(axis=1)
            combined[f"{coeff}_diff"] = abs(combined[f"{coeff}_1"] - combined[f"{coeff}_2"])
            combined[f"{coeff}_per_diff"] = round(abs(combined[f"{coeff}_diff"] / combined[f"{coeff}_mean"])*100,1)
            combined.sort_values([f"{coeff}_mean"],inplace=True,ascending=True)
            
            y = np.arange(len(combined))  # the label locations
            h = 0.4  # the width of the bars

            fig, ax = plt.subplots(figsize=(5,8))
            rects1 = ax.barh(y=y - h/2, width=combined[f"{coeff}_1"], height=h,
                            edgecolor="black", color="firebrick", label=self.label1)
            rects2 = ax.barh(y=y + h/2, width=combined[f"{coeff}_2"], height=h,
                            edgecolor="black", color="cornflowerblue", label=self.label2)
            
            # x-axis
            ax.set_xlabel(f"{coeff.title()} Value",fontsize=16)
            ax.tick_params(axis="x",labelsize=14)

            # y-axis
            ax.set_ylabel('BEVO Beacon Number',fontsize=16)
            ax.set_yticks(y)
            ax.set_yticklabels(combined["beacon"],fontsize=14)
            # remaining
            for loc in ["top","right"]:
                ax.spines[loc].set_visible(False)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1),frameon=False,ncol=2,fontsize=14)
            for beacon, (val1, val2, per) in enumerate(zip(combined[f"{coeff}_1"],combined[f"{coeff}_2"],combined[f"{coeff}_per_diff"])):
                ax.text(max(max(combined[f"{coeff}_1"]),max(combined[f"{coeff}_2"])),beacon," " + str(per) + "%",ha="left",va="center",fontsize=12)
                
            
            if save:
                plt.savefig(f"../../reports/figures/beacon_summary/calibration_comparison-{coeff}-{species}.pdf",)

            plt.show()
            plt.close()

def main():
    pass

if __name__ == '__main__':
    main()
