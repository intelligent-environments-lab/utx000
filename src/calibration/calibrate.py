
# General
import os
from datetime import datetime, timedelta
import dateutil.parser
import warnings
# Data Science
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
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
        self.date = end_time.date().strftime("%m%d%Y")

        self.data_dir = data_dir
        self.study = study
        self.suffix = study_suffix

        # kwargs
        if "resample_rate" in kwargs.keys():
            self.set_resample_rate(kwargs["resample_rate"])
        else:
            self.set_resample_rate(1) # set to default

        if "timestamp" in kwargs.keys():
            self.set_time_offset(timestamp=kwargs["timestamp"])
        else:
            self.set_time_offset() # set to default

        if "beacons" in kwargs.keys():
            self.set_beacons(kwargs["beacons"])
        else:
            self.set_beacons(np.arange(1,51,1))

        # data
        ## refererence
        print("IMPORTING REFERENCE DATA")
        self.ref = {}
        self.set_ref()
        ## beacon
        print("IMPORTING BEACON DATA")
        if self.study == "utx000":
            self.set_utx000_beacon()
        else:
            self.set_wcwh_beacon()
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
        if "timestamp" in kwargs.keys():
            self.t_offset = self.start_time - kwargs["timestamp"]
        else:
            try:
                # attempting to read pm_mass file to get the starting timestamp recorded by the computer
                temp = pd.read_csv(f"{self.data_dir}calibration/pm_mass_{self.date}.csv",skiprows=6,parse_dates={"timestamp": ["Date","Start Time"]},infer_datetime_format=True)
                self.t_offset = self.start_time - temp["timestamp"].iloc[0]
            except FileNotFoundError:
                print("No file found - try providing a `timestamp` argument instead")
                self.t_offset = 0

    def set_beacons(self, beacon_list):
        """sets the list of beacons to be considered"""
        self.beacons = beacon_list

    # reference setters
    def set_ref(self,ref_species=["pm_number","pm_mass","no2","no","co2","co"]):
        """
        Sets the reference data

        Inputs:
        ref_species: list of strings specifying the reference species data to import
        """
        for species in ref_species:
            if species in ["pm_number", "pm_mass"]:
                self.set_pm_ref(species[3:])
            elif species == "no2":
                self.set_no2_ref()
            elif species == "co2":
                self.set_co2_ref()
            elif species == "no":
                self.set_no_ref()
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

    def set_pm_ref(self, concentration_type="mass"):
        """
        Sets the reference PM data

        Inputs:
        - concentration_type: string of either "mass" or "number"

        Returns a dataframe with columns PM1, PM2.5, and PM10 indexed by timestamp
        """
        # import data and correct timestamp
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/pm_{concentration_type}_{self.date}.csv",skiprows=6)
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/pm_{concentration_type}_{self.date}.csv")
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
        df_resampled = df.resample(f"{self.resample_rate}T").mean()
        df_resampled = df_resampled[self.start_time:self.end_time]

        # setting
        for size in ["pm1","pm2p5","pm10"]:
            self.ref[f"{size}_{concentration_type}"] = pd.DataFrame(df_resampled[size]).rename(columns={size:"concentration"})
        
    def set_co2_ref(self):
        """sets the reference CO2 data"""
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/co2_{self.date}.csv",usecols=[0,1],names=["timestamp","concentration"])
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/co2_{self.date}.csv")
            return 

        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"],yearfirst=True)
        raw_data.set_index("timestamp",inplace=True)
        raw_data.index += self.t_offset# = df.shift(periods=3) 
        df = raw_data.resample(f"{self.resample_rate}T",closed="left").mean()
        self.ref["co2"] = df[self.start_time:self.end_time]

    def set_no2_ref(self):
        """sets the reference NO2 data"""
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/no2_{self.date}.csv",usecols=["IgorTime","Concentration"])
        except FileNotFoundError:
            print(f"File not found - {self.data_dir}calibration/no2_{self.date}.csv")
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

    def set_no_ref(self):
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
        
    # beacon setters
    def set_beacon_data(self,data):
        """sets the beacon data attribute with given data"""
        self.beacon_data = data

    def set_utx000_beacon(self,beacon_list=np.arange(0,51,1),verbose=False,**kwargs):
        """
        Sets beacon data from utx000 for calibration

        Inputs:
        - beacon_list: list of integers specifying the beacons to consider
        - resample_rate: integer specifying the resample rate in minutes
        - verbose: boolean to have verbose mode on
        """
        self.beacons = beacon_list
        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder=f"{self.data_dir}raw/{self.study}/beacon"
        # list of all beacons used in the study
        if verbose:
            print('Processing beacon data...\n\tReading for beacon:')
        for beacon in beacon_list:

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

                    df = pd.concat(df_list).resample(f'{self.resample_rate}T').mean() # resampling to 2 minute intervals=

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

                beacon_df.drop(['TVOC','eCO2','Visible','Infrared',"T_CO","RH_CO","T_NO2","RH_NO2",'Temperature [C]','Relative Humidity','PM_N_0p5','PM_N_4','PM_C_4'],axis=1,inplace=True)

                # removing extreme values
                #for var in beacon_df.columns:
                #    beacon_df['z'] = abs(beacon_df[var] - beacon_df[var].mean()) / beacon_df[var].std(ddof=0)
                #    beacon_df.loc[beacon_df['z'] > 3.5, var] = np.nan

                #beacon_df.drop(['z'],axis=1,inplace=True)
                # concatenating the data to the overall dataframe
                beacon_df['beacon'] = beacon
                beacon_data = pd.concat([beacon_data,beacon_df])

        beacon_data.columns = ["light","no2","co","co2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","beacon"]
        # filling in the gaps
        #beacon_data.interpolate(inplace=True)
        #beacon_data.fillna(method="bfill",inplace=True)
        self.beacon_data = beacon_data

    def set_wcwh_beacon(self, verbose=False):
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
                        temp = pd.read_csv(f"{self.data_dir}raw/{self.study}/beacon/B{number}/DATA/{file}")
                        if len(temp) > 0:
                            data_by_beacon = data_by_beacon.append(temp)
                if len(data_by_beacon) > 0:
                    data_by_beacon["Timestamp"] = pd.to_datetime(data_by_beacon["Timestamp"])
                    data_by_beacon = data_by_beacon.dropna(subset=["Timestamp"]).set_index("Timestamp").sort_index()[self.start_time:self.end_time].resample(f"{self.resample_rate}T").mean()
                    data_by_beacon["beacon"] = int(number)
                    data = data.append(data_by_beacon)
            except FileNotFoundError:
                print(f"No files found for beacon {beacon}.")
                
        data['temperature_c'] = data[['T_CO','T_NO2']].mean(axis=1)
        data['rh'] = data[['RH_CO','RH_NO2']].mean(axis=1)
        data.drop(["eCO2","Visible","Infrared","Temperature [C]","Relative Humidity","PM_N_0p5","T_CO","T_NO2","RH_CO","RH_NO2"],axis="columns",inplace=True)
        data = data[[column for column in data.columns if "1" not in column and "4" not in column]]
        data.reset_index(inplace=True)
        data.columns = ["timestamp","tvoc","lux","co","no2","pm2p5_number","pm2p5_mass","co2","beacon","temperature_c","rh"]
        data["co"] /= 1000
        self.beacon_data = data

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
            ax.set_title(beacon+1,y=1,pad=-6,loc="center",va="bottom")

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

    def compare_time_series(self,species):
        """
        Plots reference and beacon data as a time series

        Inputs:
        - species: string specifying which ieq parameter to plot
        """
        _, ax = plt.subplots(figsize=(17,6))
        ax.plot(self.ref[species].index,self.ref[species].iloc[:,0].values,linewidth=3,color="black",zorder=100,label="Reference")
        for bb in self.beacon_data["beacon"].unique():    
            data_by_bb = self.beacon_data[self.beacon_data["beacon"] == bb].set_index("timestamp")
            data_by_bb.dropna(subset=[species],inplace=True)
            if len(data_by_bb) > 0:
                ax.plot(data_by_bb.index,data_by_bb[species],marker=visualize.get_marker(int(bb)),zorder=int(bb),label=bb)
            
        # x-axis
        plt.xticks(fontsize=14,rotation=-30,ha="left")
        ax.set_xlim([self.start_time,self.end_time])
        # y_axis
        ax.set_ylabel("Concentration",fontsize=16)
        plt.yticks(fontsize=14)
        # remainder
        for loc in ["top","right"]:
            ax.spines[loc].set_visible(False)
        ax.legend(bbox_to_anchor=(1,1),frameon=False,title="Device",title_fontsize=14,fontsize=12,ncol=2)
            
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
        offsets = {"beacon":[],"mean_difference":[],"value_to_baseline":[],"correction":[]}
        ref_df = self.ref[species]

        for beacon in self.beacons:
            offsets["beacon"].append(beacon)
            # getting relevant data
            beacon_df = self.beacon_data[self.beacon_data["beacon"] == beacon].set_index("timestamp")
            beacon_df.dropna(subset=[species],inplace=True)
            if len(beacon_df) > 1:
                # merging beacon and reference data to get difference
                df = pd.merge(left=beacon_df,left_index=True,right=ref_df,right_index=True,how="inner")
                df["delta"] = df[species] - df["concentration"]
                # adding data
                mean_delta = np.nanmean(df["delta"])
                val_to_base = np.nanmin(df[species]) - baseline
                offsets["mean_difference"].append(mean_delta)
                offsets["value_to_baseline"].append(val_to_base)
                if np.nanmin(df[species]) - mean_delta < baseline:
                    offsets["correction"].append(np.nanmin(df[species]) - baseline)
                else:
                    offsets["correction"].append(mean_delta)     
            else:
                # adding zeros
                offsets["mean_difference"].append(0)
                offsets["value_to_baseline"].append(0)
                offsets["correction"].append(0)

        offset_df = pd.DataFrame(data=offsets)
        offset_df.set_index("beacon",inplace=True)
        self.offsets[species] = offset_df
        if save_to_file:
            s = species.lower() #string of variable
            offset_df.to_csv(f"{self.data_dir}interim/{s}-offset-{self.suffix}.csv")

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

                    beacon_df[species] -= offset_df.loc[i,"correction"]
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
                    beacon_df[species] -= offset_df.loc[bb,"correction"]
                    ax.plot(beacon_df.index,beacon_df[species],marker=self.get_marker(int(bb)),zorder=int(bb),label=bb)
            
            for spine in ["top","right"]:
                ax.spines[spine].set_visible(False) 

            ax.plot(ref_df.index,ref_df["concentration"],color="black",zorder=99)
            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()

    def linear_regression(self,ref_data,beacon_data,ref_var,beacon_var,save_to_file=False,show_plot=False,show_corrected=False,verbose=False):
        """
        Runs a linear regression model
        
        Inputs:
        - ref_data: dataframe of reference data with single column corresponding to data indexed by time
        - beacon_data: dataframe of beacon data with two columns corresponding to data and beacon number indexed by time
        - test_size: float specifying the proportion of data to use for the training set
        - show_plot: boolean to show the plot or not

        Returns coefficient(s) of the linear fit
        """
        coeffs = {"beacon":[],"constant":[],"coefficient":[]}
        ref_df = ref_data[ref_var]
        beacon_data = beacon_data[[beacon_var,"beacon"]]
        for bb in np.arange(1,51):
            beacon_by_bb = beacon_data[beacon_data["beacon"] == bb]
            if verbose:
                print(f"Working for Beacon {bb}")
                print(beacon_by_bb.head())
            if len(beacon_by_bb) > 1:
                beacon_by_bb.drop(["beacon"],axis=1,inplace=True)
                if len(ref_data) == len(beacon_by_bb):
                    pass
                    #index = int(test_size*len(ref_data))
                else:
                    # resizing arrays to included data from both modalities
                    max_start_date = max(ref_df.index[0],beacon_by_bb.index[0])
                    min_end_date = min(ref_df.index[-1],beacon_by_bb.index[-1])
                    ref_df = ref_df[max_start_date:min_end_date]
                    beacon_by_bb = beacon_by_bb[max_start_date:min_end_date]
                    warnings.warn("Reference and beacon data are not the same length")

                df = pd.merge(left=ref_df,right=beacon_by_bb,left_index=True,right_index=True,how="inner")
                if verbose:
                    print(df.head())

                if len(df) > 2:
                    df.dropna(inplace=True)

                    times = []
                    for t in df.index:
                        times.append((t - beacon_by_bb.index[0]).total_seconds()/60)

                    y = df.loc[:,"concentration"].values
                    x = df.loc[:,beacon_var].values

                    # linear regression model
                    regr = linear_model.LinearRegression()
                    try:
                        regr.fit(x.reshape(-1, 1), y)
                    except ValueError:
                        print("Error with data.")
                        continue

                    # adding data
                    coeffs["beacon"].append(bb)
                    coeffs["constant"].append(regr.intercept_)
                    coeffs["coefficient"].append(regr.coef_[0])

                    # plotting
                    if show_plot == True:
                        fig, ax = plt.subplots(figsize=(6,6))
                        im = ax.scatter(x,y,c=times,cmap="Blues",edgecolor="black",s=75,label="Measured",zorder=2)
                        fig.colorbar(im,ax=ax,label="Minutes since Start")
                        # Make draw the line of best-fit
                        y_pred = regr.predict(x.reshape(-1, 1))
                        ax.plot(x,y_pred,color='firebrick',linewidth=3,label="Prediction",zorder=3)
                        ax.legend(bbox_to_anchor=(1.65,1),frameon=False)

                        #plt_min = min(min(x),min(y))
                        #plt_max = max(max(x),max(y))
                        #ax.text(0.975*plt_min,0.975*plt_min,f"Coefficient: {round(regr.coef_[0],3)}",backgroundcolor="white",ma="center")
                        #ax.set_xlim([0.95*plt_min,1.05*plt_max])
                        ax.set_xlabel("Beacon Measurement")
                        #ax.set_ylim([0.95*plt_min,1.05*plt_max])
                        ax.set_ylabel("Reference Measurement")
                        ax.set_title(bb)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)

                        plt.show()
                        plt.close()

                else:
                    warnings.warn("Merged dataframe has no length - check data availability and timestamps")
            else:
                # adding blank
                coeffs["beacon"].append(bb)
                coeffs["constant"].append(0)
                coeffs["coefficient"].append(1)

        coeff_df = pd.DataFrame(coeffs)
        coeff_df.set_index("beacon",inplace=True)

        if show_corrected:
            _, ax = plt.subplots(figsize=(16,6))
            for bb in beacon_data["beacon"].unique():
                beacon_df = beacon_data[beacon_data["beacon"] == bb]
                beacon_df.dropna(subset=[beacon_var],inplace=True)
                if verbose:
                    print(beacon_df.head())
                if len(beacon_df) > 1:
                    beacon_df[beacon_var] = beacon_df[beacon_var] * coeff_df.loc[bb,"coefficient"] + coeff_df.loc[bb,"constant"]
                    ax.scatter(beacon_df.index,beacon_df[beacon_var],marker=self.get_marker(int(bb)),zorder=int(bb),label=bb)
            
            for spine in ["top","right"]:
                ax.spines[spine].set_visible(False) 

            ax.plot(ref_df.index,ref_df["concentration"],color="black",linewidth=3,zorder=99)
            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()

        if save_to_file:
                s = beacon_var.lower() #string of variable
                coeff_df.to_csv(f"{self.data_dir}interim/{s}-linear_model-{self.suffix}.csv")

        return coeff_df

    # saving
    def save_offsets():
        """saves offset results to file"""
        pass

    def save_lms():
        """saves linear model results to file"""
        pass

def main():
    pass

if __name__ == '__main__':
    main()
