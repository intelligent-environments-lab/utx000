
# General
import os
import math
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class Calibration():

    def __init__(self, start_time, end_time, data_dir="../../data/", study_suffix="ux_s20", calibration_start=datetime(2020,12,30,12,40,0)):
        """
        Initiates the calibration object with:
        - start_time: datetime object with precision to the minute specifying the event START time
        - end_time: datetime object with precision to the minute specifying the event END time
        - data_dir: path to data directory
        """
        self.start_time = start_time
        self.end_time = end_time
        self.date = end_time.date().strftime("%m%d%Y")
        self.calibration_start = calibration_start # start datetime for the beginning of ALL calibration events

        self.data_dir = data_dir
        self.suffix = study_suffix

    def get_pm_ref(self,file,resample_rate=2):
        """
        Gets the reference PM data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with columns PM1, PM2.5, and PM10 indexed by timestamp
        """
        raw_data = pd.read_csv(f"{self.data_dir}calibration/"+file,skiprows=6)
        df = raw_data.drop(['Sample #','Aerodynamic Diameter'],axis=1)
        date = df['Date']
        sample_time = df['Start Time']
        datetimes = []
        for i in range(len(date)):
            datetimes.append(datetime.strptime(date[i] + ' ' + sample_time[i],'%m/%d/%y %H:%M:%S') + timedelta(minutes=4))

        df['Timestamp'] = datetimes
        df = df.set_index(['Timestamp'])
        df = df.iloc[:,:54]
        df = df.drop(['Date','Start Time'],axis=1)

        for column in df.columns:
            df[column] = pd.to_numeric(df[column])

        if file[3:16] == "concentration":
            factor = 1000
        else:
            factor = 1

        df['PM_1'] = df.iloc[:,:10].sum(axis=1)*factor
        df['PM_2p5'] = df.iloc[:,:23].sum(axis=1)*factor
        df['PM_10'] = df.iloc[:,:42].sum(axis=1)*factor

        df_resampled = df.resample(f"{resample_rate}T").mean()
        return df_resampled[self.start_time:self.end_time]

    def get_co2_ref(self,file,resample_rate=2):
        """
        Gets the reference CO2 data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with co2 concentration data indexed by time
        """
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/{file}",usecols=[0,1],names=["Timestamp","Concentration"])
            raw_data["Timestamp"] = pd.to_datetime(raw_data["Timestamp"],yearfirst=True)
            raw_data.set_index("Timestamp",inplace=True)
            df = raw_data.resample(f"{resample_rate}T").mean()
            return df[self.start_time:self.end_time]
        except FileNotFoundError:
            print("No file found for this event - returning empty dataframe")
            return pd.DataFrame()

    def get_no2_ref(self,file,resample_rate=2):
        """
        Gets the reference NO2 data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with no2 concentration data indexed by time
        """
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/{file}",usecols=["IgorTime","Concentration"])
            # Using igor time (time since Jan 1st, 1904) to get timestamp
            ts = []
            for seconds in raw_data["IgorTime"]:
                ts.append(datetime(1904,1,1) + timedelta(seconds=int(seconds)))
            raw_data["Timestamp"] = ts
            raw_data.set_index("Timestamp",inplace=True)
            raw_data.drop("IgorTime",axis=1,inplace=True)

            df = raw_data.resample(f"{resample_rate}T").mean()
            return df[self.start_time:self.end_time]
        except FileNotFoundError:
            print("No file found for this event - returning empty dataframe")
            return pd.DataFrame()

    def get_no_ref(self,file,resample_rate=2):
        """
        Gets the reference no data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with no concentration data indexed by time
        """
        try:
            raw_data = pd.read_csv(f"{self.data_dir}calibration/{file}",names=["TimeStamp","Concentration"],skiprows=1,index_col=0,parse_dates=True,infer_datetime_format=True)
            df = raw_data.resample(f"{resample_rate}T").mean()
            return df[self.start_time:self.end_time]
        except FileNotFoundError:
            print("No file found for this event - returning empty dataframe")
            return pd.DataFrame()

    def get_beacon_data(self,beacon_list=np.arange(0,51,1),resample_rate=2,verbose=False,**kwargs):
        """
        Gets beacon data to calibrate

        Inputs:
        - beacon_list: list of integers specifying the beacons to consider
        - resample_rate: integer specifying the resample rate in minutes
        - verbose: boolean to have verbose mode on

        Returns a dataframe with beacon measurements data indexed by time and a column specifying the beacon number
        """
        self.beacons = beacon_list
        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder=f"{self.data_dir}raw/bpeace2/beacon"
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

                        except Exception as inst:
                            # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                            if verbose:
                                print(f'\t\tIssue encountered while importing {csv_dir}/{file}, skipping...')

                    df = pd.concat(df_list).resample(f'{resample_rate}T').mean() # resampling to 2 minute intervals=

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

                beacon_df.drop(['TVOC', 'eCO2', 'Lux', 'Visible', 'Infrared', "CO","T_CO","RH_CO","T_NO2","RH_NO2",'Temperature [C]','Relative Humidity','PM_N_4','PM_C_4'],axis=1,inplace=True)

                # concatenating the data to the overall dataframe
                beacon_df['Beacon'] = beacon
                beacon_data = pd.concat([beacon_data,beacon_df])

        return beacon_data

    def inspect(self,df,timeseries=True):
        """
        Visually inspect data in dataframe

        Inputs:
        - df: dataframe with one column with values or column named "Beacons" that includes the beacon number
        - timeseries: boolean specifying whether or not to plot the timeseries or not (therefore heatmap)
        """

        if timeseries:
            fig, ax = plt.subplots(figsize=(16,6))
            if "Beacon" in df.columns:
                for bb in df["Beacon"].unique():
                    df_by_bb = df[df["Beacon"] == bb]
                    ax.plot(df_by_bb.index,df_by_bb.iloc[:,0].values,marker=self.get_marker(int(bb)),label=bb)
            else:
                ax.plot(df.index,df.iloc[:,0].values,linewidth=3,color="black",label="Ref")

            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()
        else: #heatmap
            fig,ax = plt.subplots(figsize=(14,7))
            if "Beacon" in df.columns:
                df.columns=["Concentration","Beacon"]
                df_to_plot = pd.DataFrame()
                for bb in df["Beacon"].unique():
                    df_by_bb = df[df["Beacon"] == bb]
                    df_by_bb.drop("Beacon",axis=1,inplace=True)
                    df_to_plot = pd.concat([df_to_plot,df_by_bb],axis=1)
                    df_to_plot.rename(columns={"Concentration":bb}, inplace=True)

                sns.heatmap(df_to_plot.T,vmin=np.nanmin(df_to_plot),vmax=np.nanmax(df_to_plot),ax=ax)
                locs, labels = plt.xticks()
                new_labels = []
                for label in labels:
                    new_labels.append(dateutil.parser.isoparse(label.get_text()).strftime("%m-%d-%y %H:%M"))
                plt.xticks(locs,new_labels,rotation=-45,ha="left")
                plt.yticks(rotation=0,va="center")
            else:
                sns.heatmap(df.T,vmin=np.nanmin(df),vmax=np.nanmax(df),ax=ax)

    def compare_time_series(self,ref_data,beacon_data):
        """
        Plots reference and beacon data as a time series

        Inputs:
        - ref_data: dataframe of reference data with single column corresponding to data indexed by time
        - beacon_data: dataframe of beacon data with two columns corresponding to data and beacon number indexed by time
        """
        ref_data = ref_data[self.start_time:self.end_time]
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(ref_data.index,ref_data.iloc[:,0].values,linewidth=3,color="black",zorder=100,label="Reference")
        for bb in beacon_data.iloc[:,1].unique():    
            data_by_bb = beacon_data[beacon_data.iloc[:,1] == bb]
            data_by_bb = data_by_bb[self.start_time:self.end_time]
            try:
                data_by_bb.drop("Beacon",axis=1,inplace=True)
            except KeyError:
                data_by_bb.drop("beacon",axis=1,inplace=True)

            data_by_bb.dropna(inplace=True)
            
            if len(data_by_bb) > 0:
                ax.plot(data_by_bb.index,data_by_bb.iloc[:,0].values,marker=self.get_marker(int(bb)),zorder=int(bb),label=bb)
            
        ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            
        plt.show()
        plt.close()

    def compare_histogram(self,ref_data,beacon_data,bins):
        """
        Plots reference and beacon data as histograms

        Inputs:
        - ref_data: dataframe of reference data with single column corresponding to data indexed by time
        - beacon_data: dataframe of beacon data with two columns corresponding to data and beacon number indexed by time
        """
        fig, ax = plt.subplots(10,5,figsize=(30,15),sharex=True)  
        for i, axes in enumerate(ax.flat):
            # getting relevant data
            beacon_df = beacon_data[beacon_data["Beacon"] == i]
            beacon_df.dropna(inplace=True)
            if len(beacon_df) > 1:
                # reference data
                axes.hist(ref_data.iloc[:,0].values,bins=bins,color="black",zorder=1,alpha=0.7)
                # beacon data
                axes.hist(beacon_df.iloc[:,0].values,bins=bins,color="seagreen",zorder=9) 
                axes.set_title(f"Beacon {i}")          
            else:
                # making it easier to read by removing the unused figures
                axes.set_xticks([])
                axes.set_yticks([])
                for spine in ["top","right","bottom","left"]:
                    axes.spines[spine].set_visible(False)  
            
        plt.show()
        plt.close()

    def get_reporting_beacons(self,beacon_data,beacon_var,beacon_col="Beacon"):
        """
        Gets the list of beacons that report measurements from the specified sensor
        """
        var_only = beacon_data[[beacon_var,beacon_col]]
        reporting_beacons = []
        for bb in var_only[beacon_col].unique():
            df = var_only.dropna(subset=[beacon_var])
            if len(df) > 2:
                reporting_beacons.append(bb)
        try:
            if beacon_var.lower() == "no2":
                possible_beacons = [x for x in self.beacons if x <= 28] # getting no2 sensing beacons only
                missing_beacons = [x for x in possible_beacons if x not in reporting_beacons]
            else:
                missing_beacons = [x for x in self.beacons if x not in reporting_beacons]

            print(f"Missing data from: {missing_beacons}")
        except AttributeError:
            print("Calibration object has no attribute beacons - need to run get_beacon_data")
            return [], reporting_beacons

        return missing_beacons, reporting_beacons

    def get_marker(self,number):
        """
        Gets a marker style based on the beacon number
        """
        if number < 10:
            m = "s"
        elif number < 20:
            m = "^"
        elif number < 30:
            m = "P"
        elif number <40:
            m = "*"
        else:
            m = "o"

        return m

    def offset(self,ref_data,beacon_data,ref_var,beacon_var,groups=[],save_to_file=False,show_corrected=False):
        """
        Gets the average offset value and standard deviation between the beacon and reference measurement

        Inputs:
        - ref_data: dataframe holding the reference data
        - beacon_data: dataframe holding the beacon data with column corresponding to the beacon number
        - ref_var: string of the reference variable in the ref dataframe
        - beacon_var: string of the beacon variable in the beacond dataframe
        - groups: list of list for any groups that should be highlighted in the figure

        Returns dataframe holding the average difference and standard deviation between the differences
        """
        offsets = {"beacon":[],"mean":[],"std":[]}
        ref_df = ref_data[ref_var]

        colors = ["seagreen","cornflowerblue","firebrick","goldenrod"]
        fig, ax = plt.subplots(10,5,figsize=(30,15),sharex=True)  
        for i, axes in enumerate(ax.flat):
            offsets["beacon"].append(i)
            # getting relevant data
            beacon_df = beacon_data[beacon_data["Beacon"] == i]
            beacon_df.dropna(subset=[beacon_var],inplace=True)
            if len(beacon_df) > 1:
                # merging beacon and reference data to get difference
                df = pd.merge(left=beacon_df,left_index=True,right=ref_df,right_index=True,how="inner")
                df["delta"] = df[beacon_var] - df["Concentration"]
                # adding data
                offsets["mean"].append(np.nanmean(df["delta"]))
                offsets["std"].append(np.nanstd(df["delta"]))
                axes.scatter(df.index,df["delta"],s=9,color="black") # everything is plotted in black
                for j, group in enumerate(groups):
                    if i in group:
                        axes.scatter(df.index,df["delta"],s=10,color=colors[j]) # this will plot over the initial black scatter

                axes.set_title(f"Beacon {i}")  
                for spine in ["top","right","bottom"]:
                    axes.spines[spine].set_visible(False)        
            else:
                # adding zeros
                offsets["mean"].append(np.nanmean(0))
                offsets["std"].append(np.nanstd(0))
                # making it easier to read by removing the unused figures
                axes.set_xticks([])
                axes.set_yticks([])
                for spine in ["top","right","bottom","left"]:
                    axes.spines[spine].set_visible(False)    

        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=-30,ha="left")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
        plt.close()

        offset_df = pd.DataFrame(data=offsets)
        offset_df.set_index("beacon",inplace=True)
        if save_to_file:
            s = beacon_var.lower() #string of variable
            offset_df.to_csv(f"{self.data_dir}interim/{s}-offset-{self.suffix}.csv")

        if show_corrected:
            # Plotting Corrected Timeseries by Beacon
            # ---------------------------------------
            fig, ax = plt.subplots(10,5,figsize=(30,15),sharex=True)  
            for i, axes in enumerate(ax.flat):
                # getting relevant data
                beacon_df = beacon_data[beacon_data["Beacon"] == i]
                beacon_df.dropna(subset=[beacon_var],inplace=True)
                if len(beacon_df) > 1:
                    axes.plot(ref_df.index,ref_df["Concentration"],color="black")

                    beacon_df[beacon_var] -= offset_df.loc[i,"mean"]
                    axes.plot(beacon_df.index,beacon_df[beacon_var],color="seagreen")
                    axes.set_title(f"Beacon {i}")  
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
            all_beacon = self.get_beacon_data(start_time=self.calibration_start)
            fig, ax = plt.subplots(figsize=(16,6))
            for bb in all_beacon["Beacon"].unique():
                beacon_df = all_beacon[all_beacon["Beacon"] == bb]
                beacon_df.dropna(subset=[beacon_var],inplace=True)
                if len(beacon_df) > 1:
                    beacon_df[beacon_var] -= offset_df.loc[bb,"mean"]
                    ax.plot(beacon_df.index,beacon_df[beacon_var],marker=self.get_marker(int(bb)),zorder=int(bb),label=bb)
            
            for spine in ["top","right"]:
                ax.spines[spine].set_visible(False) 

            ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            plt.show()
            plt.close()

        return offset_df

    def linear_regression(self,ref_data,beacon_data,test_size=1,show_plot=True):
        """
        Runs a regression model
        
        Inputs:
        - ref_data: dataframe of reference data with single column corresponding to data indexed by time
        - beacon_data: dataframe of beacon data with two columns corresponding to data and beacon number indexed by time
        - test_size: float specifying the proportion of data to use for the training set
        - show_plot: boolean to show the plot or not

        Returns coefficient(s) of the linear fit
        """

        if len(ref_data) == len(beacon_data):
            index = int(test_size*len(ref_data))
        else:
            # resizing arrays to included data from both modalities
            max_start_date = max(ref_data.index[0],beacon_data.index[0])
            min_end_date = min(ref_data.index[-1],beacon_data.index[-1])
            ref_data = ref_data[max_start_date:min_end_date]
            beacon_data = beacon_data[max_start_date:min_end_date]
            warnings.warn("Reference and Beacon data are not the same length")
        # splitting beacon data
        if test_size == 1:
            times = []
            for t in beacon_data.index:
                times.append((t - beacon_data.index[0]).total_seconds()/60)

            y_train = beacon_data.iloc[:,0].values
            y_test = beacon_data.iloc[:,0].values
        else:
            times = []
            for t in beacon_data.index[:index]:
                times.append((t - beacon_data.index[0]).total_seconds()/60)
            y_train = beacon_data.iloc[:,0].values[:index]
            y_test = beacon_data.iloc[:,0].values[index:]

        # splitting ref data
        if test_size == 1:
            x_train = ref_data.iloc[:,0].values
            x_test = ref_data.iloc[:,0].values
        else:
            x_train = ref_data.iloc[:,0].values[:index]
            x_test = ref_data.iloc[:,0].values[index:]

        # linear regression model
        regr = linear_model.LinearRegression()
        regr.fit(x_train.reshape(-1, 1), y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(x_test.reshape(-1, 1))

        # plotting
        if show_plot == True:
            fig, ax = plt.subplots(figsize=(6,6))
            im = ax.scatter(x_train,y_train,c=times,cmap="Blues",edgecolor="black",s=75,label="Training",zorder=2)
            fig.colorbar(im,ax=ax,label="Minutes since Start")
            ax.scatter(x_test,y_test,color='seagreen',label="Test",zorder=1)
            ax.plot(x_test,y_pred,color='firebrick',linewidth=3,label="Prediction",zorder=3)
            ax.legend(bbox_to_anchor=(1.65,1),frameon=False)

            plt_min = min(min(x_train),min(y_train))
            plt_max = max(max(x_train),max(y_train))
            #ax.text(0.975*plt_min,0.975*plt_min,f"Coefficient: {round(regr.coef_[0],3)}",backgroundcolor="white",ma="center")
            #ax.set_xlim([0.95*plt_min,1.05*plt_max])
            ax.set_xlabel("Reference Measurement")
            #ax.set_ylim([0.95*plt_min,1.05*plt_max])
            ax.set_ylabel("Beacon Measurement")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            plt.show()
            plt.close()
            
        return regr.coef_

def main():
    pass

if __name__ == '__main__':
    main()
