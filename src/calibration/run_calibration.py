import os

import pandas as pd
import numpy as np

from datetime import datetime, timedelta

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
            datetimes.append(datetime.strptime(date[i] + ' ' + sample_time[i],'%m/%d/%y %H:%M:%S'))

        df['Timestamp'] = datetimes
        df = df.set_index(['Timestamp'])
        df = df.iloc[:,:54]
        df = df.drop(['Date','Start Time'],axis=1)

        for column in df.columns:
            df[column] = pd.to_numeric(df[column])

        df['PM_1'] = df.iloc[:,:10].sum(axis=1)*1000
        df['PM_2p5'] = df.iloc[:,:23].sum(axis=1)*1000
        df['PM_10'] = df.iloc[:,:42].sum(axis=1)*1000

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
        raw_data = pd.read_csv(f"{self.data_dir}calibration/{file}",usecols=[0,1],names=["Timestamp","Concentration"])
        raw_data["Timestamp"] = pd.to_datetime(raw_data["Timestamp"],yearfirst=True)
        raw_data.set_index("Timestamp",inplace=True)
        df = raw_data.resample(f"{resample_rate}T").mean()
        return df[self.start_time:self.end_time]

    def get_no2_ref(self,file,resample_rate=2):
        """
        Gets the reference NO2 data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with no2 concentration data indexed by time
        """
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

    def get_no_ref(self,file,resample_rate=2):
        """
        Gets the reference no data

        Inputs:
        - file: string holding the reference data location name
        - resample_rate: integer specifying the resample rate in minutes

        Returns a dataframe with no concentration data indexed by time
        """
        raw_data = pd.read_csv(f"{self.data_dir}calibration/{file}",names=["TimeStamp","Concentration"],skiprows=1,index_col=0,parse_dates=True,infer_datetime_format=True)
        df = raw_data.resample(f"{resample_rate}T").mean()
        return df[self.start_time:self.end_time]

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

    def plot_time_series(ref_data,beacon_data):
        """
        Plots reference and beacon data as a time series
        """
        
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(ref_data.index,ref_data.iloc[:,0].values,linewidth=3,color="black",zorder=100)
        for bb in beacon_data["Beacon"].unique():
            if bb < 10:
                m = "s"
            elif bb < 20:
                m = "^"
            elif bb < 30:
                m = "P"
            elif bb <40:
                m = "*"
            else:
                m = "o"
                
            data_by_bb = beacon_data[beacon_data["Beacon"] == bb]
            data_by_bb.drop("Beacon",axis=1,inplace=True)
            data_by_bb.dropna(inplace=True)
            
            if len(data_by_bb) > 0:
                ax.plot(data_by_bb.index,data_by_bb.iloc[:,0].values,marker=m,label=bb)
            
        ax.legend(bbox_to_anchor=(1,1))
            
        plt.show()
        plt.close()

class Linear_Model(Calibration):

    def __init__(self):
        """

        """

def main():
    pass

if __name__ == '__main__':
    main()
