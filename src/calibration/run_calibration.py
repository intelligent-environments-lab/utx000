
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

    def __init__(self, start_time, end_time, data_dir="../../data/"):
        """
        Initiates the calibration object with:
        - start_time: datetime object with precision to the minute specifying the event START time
        - end_time: datetime object with precision to the minute specifying the event END time
        - data_dir: path to data directory
        """
        self.start_time = start_time
        self.end_time = end_time
        self.date = start_time.date().strftime("%m%d%Y")

        self.data_dir = data_dir

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

        if file[3:] == "concentration":
            factor = 1000

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

    def get_beacon_data(self,beacon_list=np.arange(0,51,1),resample_rate=2,verbose=False):
        """
        Gets beacon data to calibrate

        Inputs:
        - beacon_list: list of integers specifying the beacons to consider
        - resample_rate: integer specifying the resample rate in minutes
        - verbose: boolean to have verbose mode on

        Returns a dataframe with beacon measurements data indexed by time and a column specifying the beacon number
        """
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
        
        fig, ax = plt.subplots(figsize=(16,6))
        ax.plot(ref_data.index,ref_data.iloc[:,0].values,linewidth=3,color="black",zorder=100)
        for bb in beacon_data["Beacon"].unique():    
            data_by_bb = beacon_data[beacon_data["Beacon"] == bb]
            data_by_bb.drop("Beacon",axis=1,inplace=True)
            data_by_bb.dropna(inplace=True)
            
            if len(data_by_bb) > 0:
                ax.plot(data_by_bb.index,data_by_bb.iloc[:,0].values,marker=self.get_marker(int(bb)),label=bb)
            
        ax.legend(bbox_to_anchor=(1,1),frameon=False,ncol=2)
            
        plt.show()
        plt.close()

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

class Linear_Model(Calibration):

    def __init__(self,start_time,end_time,data_dir):
        """
        Initializes Linear Model pulling from the Calibration Class
        """
        super().__init__(start_time,end_time,data_dir)
        self.model_type="Linear"

    def regression(self,ref_data,beacon_data,test_size=1,show_plot=True):
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
            y_train = beacon_data.iloc[:,0].values
            y_test = beacon_data.iloc[:,0].values
        else:
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
            ax.scatter(x_train,y_train,color="orange",alpha=0.7,label="Training")
            ax.scatter(x_test,y_test,color='seagreen',label="Test")
            ax.plot(x_test,y_pred,color='cornflowerblue',linewidth=3,label="Prediction")
            ax.legend(bbox_to_anchor=(1,1),frameon=False)

            plt_min = min(min(x_train),min(y_train))
            plt_max = max(max(x_train),max(y_train))
            ax.text(0.975*plt_min,0.975*plt_min,f"Coefficient: {round(regr.coef_[0],3)}")
            ax.set_xlim([0.95*plt_min,1.05*plt_max])
            ax.set_ylim([0.95*plt_min,1.05*plt_max])

            plt.show()
            plt.close()
            
        return regr.coef_

def main():
    pass

if __name__ == '__main__':
    main()
