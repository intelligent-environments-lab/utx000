# -*- coding: utf-8 -*-
# Data Science Packages
from numpy.lib.twodim_base import vander
import pandas as pd
import numpy as np
import scipy.stats as stats

# Useful
from datetime import datetime, timedelta
import math

# Operations
import os
import os.path
import logging

# Extra
import ast
import warnings
warnings.filterwarnings('ignore')

# Local
from ut3000 import ut1000, ut2000, ut3000
from wcwh import wcwh

class bpeace():
    '''
    Class used to process bpeace data (Spring 2020)
    '''

    def __init__(self):
        self.study = 'bpeace'
        self.id_crossover = pd.read_excel('../../data/raw/bpeace/admin/id_crossover.xlsx',sheet_name='id')
        self.beacon_id = pd.read_excel('../../data/raw/bpeace/admin/id_crossover.xlsx',sheet_name='beacon')

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def process_beacon(self):
        '''
        Combines data from all sensors on all beacons

        Returns True if able to save one dataframe that contains all the data at regular intervals in /data/processed directory
        '''

        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        # list of all beacons used in the study
        beacon_list = [29,28,27,26,25,24,23,22,20,19,18,17,16,15,14,13,12,11,10,9,7,6,5,4,3,2,1]
        print('\tProcessing beacon data...\n\t\tReading for beacon:')
        for beacon in beacon_list:
            print(f'\t\t{beacon}')
            beacon_df = pd.DataFrame() # dataframe specific to the beacon
            # correcting the number since the values <10 have leading zero in directory
            if beacon < 10:
                number = f'0{beacon}'
            else:
                number = f'{beacon}'
            # getting other ids
            for i in range(len(self.id_crossover)):
                if beacon == self.id_crossover['Beacon'][i]:
                    beiwe = self.id_crossover['Beiwe'][i]
                    fitbit = self.id_crossover['Fitbit'][i]
                    redcap = self.id_crossover['REDCap'][i]

            # Python3 Sensors
            # ---------------
            py3_df = pd.DataFrame() # dataframe for sensors using python3
            for file in os.listdir(f'../../data/raw/bpeace/beacon/B{number}/adafruit/'):
                try:
                    # reading in raw data (csv for one day at a time) and appending it to the overal dataframe
                    day_df = pd.read_csv(f'../../data/raw/bpeace/beacon/B{number}/adafruit/{file}',
                                        index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py3_df = pd.concat([py3_df,day_df])
                except Exception as inst:
                    # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'../../data/raw/bpeace/beacon/B{number}/adafruit/{file}',f'../../data/purgatory/{self.study}-B{number}-py3-{file}')

            py3_df = py3_df.resample('5T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)
            # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues - see Hagen)
            if number in ['28','29']:
                print('\t\t\tNo NO2 sensor - removing values')
                py3_df['CO'] = py3_df['NO2']
                py3_df['NO2'] = np.nan
                py3_df['T_CO'] = py3_df['T_NO2']
                py3_df['T_NO2'] = np.nan
                py3_df['RH_CO'] = py3_df['RH_NO2']
                py3_df['RH_NO2'] = np.nan
            py3_df['CO'] /= 1000 # converting ppb measurements to ppm

            # Python2 Sensors
            # ---------------
            py2_df = pd.DataFrame()
            for file in os.listdir(f'../../data/raw/bpeace/beacon/B{number}/sensirion/'):
                try:
                    day_df = pd.read_csv(f'../../data/raw/bpeace/beacon/B{number}/sensirion/{file}',
                                    index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py2_df = pd.concat([py2_df,day_df])
                except Exception as inst:
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'../../data/raw/bpeace/beacon/B{number}/sensirion/{file}',f'../../data/purgatory/{self.study}-B{number}-py2-{file}')
                
            for col in py2_df.columns:
                py2_df[col] = pd.to_numeric(py2_df[col],errors='coerce')

            py2_df = py2_df.resample('5T').mean()
                
            # merging python2 and 3 sensor dataframes
            beacon_df = py3_df.merge(right=py2_df,left_index=True,right_index=True,how='outer')
            # getting relevant data only
            start_date = self.beacon_id[self.beacon_id['Beiwe'] == beiwe]['start_date'].values[0]
            end_date = self.beacon_id[self.beacon_id['Beiwe'] == beiwe]['end_date'].values[0]
            beacon_df = beacon_df[start_date:end_date]
            # removing bad values from important variables
            important_vars = ['TVOC','CO2','NO2','CO','PM_C_2p5','PM_C_10','T_NO2','T_CO','Temperature [C]','Lux','RH_NO2','RH_CO','Relative Humidity']
            # variables that should never have anything less than zero
            for var in ['CO2','T_NO2','T_CO','Temperature [C]','RH_NO2','RH_CO','Relative Humidity']:
                beacon_df[var].mask(beacon_df[var] < 0, np.nan, inplace=True)
            # variables that should never be less than a certain limit
            for var, threshold in zip(['CO2','Lux'],[100,-1]):
                beacon_df[var].mask(beacon_df[var] < threshold, np.nan, inplace=True)
            # removing extreme values (zscore greater than 2.5)
            for var in important_vars:
                beacon_df['z'] = abs(beacon_df[var] - np.nanmean(beacon_df[var])) / np.nanstd(beacon_df[var])
                beacon_df.loc[beacon_df['z'] > 2.5, var] = np.nan
            # adding columns for the pt details
            beacon_df['Beacon'] = beacon
            beacon_df['Beiwe'] = beiwe
            beacon_df['Fitbit'] = fitbit
            beacon_df['REDCap'] = redcap
            
            beacon_data = pd.concat([beacon_data,beacon_df])

        # saving
        try:
            beacon_data.to_csv(f'../../data/processed/bpeace-beacon.csv')
        except:
            return False

        return True

    def process_weekly_surveys(self):
        '''
        Processes raw weekly survey answers. The survey IDs are:
        - eQ2L3J08ChlsdSXXKOoOjyLJ: morning
        - 7TaT8zapOWO0xdtONnsY8CE0: evening
        
        Returns True if able to save two dataframes for morning/evening survey data in /data/processed directory
        '''
        # defining some variables for ease of understanding
        parent_dir = '../../data/raw/bpeace/beiwe/survey_answers/'
        morning_survey_id = 'vBewaVfZ6oWcsiAoPvF6CZi7'
        evening_survey_id = 'OymqfwTdyaHFIsJoUNIfPWyG'
        weekly_survey_id = 'aMIwBMFUgO8Rtw2ZFjyMTzDn'
        
        # defining the final dataframes to append to
        evening_survey_df = pd.DataFrame()
        morning_survey_df = pd.DataFrame()
        weekly_survey_df = pd.DataFrame()
        
        # Morning Survey Data
        # -------------------
        print('\tProcessing morning survey data...')
        # looping through the participants and then all their data
        for participant in os.listdir(parent_dir):
            # making sure we don't read from any hidden directories/files
            if len(participant) == 8:
                pid = participant
                participant_df = pd.DataFrame(columns=['beiwe','content','stress','lonely','sad','energy','tst','sol','naw','restful'])
            
                for file in os.listdir(f'{parent_dir}{participant}/survey_answers/{morning_survey_id}/'):
                    # reading raw data
                    df = pd.read_csv(f'{parent_dir}{participant}/survey_answers/{morning_survey_id}/{file}')
                    # adding new row
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
                                                                                               df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with morning survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{parent_dir}{participant}/survey_answers/{morning_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-morning-{file}')
            
                # appending participant df to overall df
                morning_survey_df = morning_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
        
        # replacing string values with numeric
        morning_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        morning_survey_df.replace({'6-7 hours':6.5,'9-10 hours':9.5,'5-6 hours':5.5,'8-9 hours':8.5,'7-8 hours':7.5,'2-3 hours':2.5,
                                '0 hours; did not sleep':0,'4-5 hours':4.5,'3-4 hours':3.5,'more than 12 hours':12,
                                '1-2 hours':1.5,'11-12 hours':11.5,'6':6,'5':5,'7':7,'8':8,'-1':np.nan,'10-11 hours':10.5,'4':4,'9':9,'3':3,'1':1,'2':2},inplace=True)
        # fixing any string inputs outside the above range
        morning_survey_df['naw'] = pd.to_numeric(morning_survey_df['naw'],errors='coerce')
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(parent_dir):
            if len(participant) == 8:
                pid = participant
                # pre-allocating dataframe columns
                participant_df = pd.DataFrame(columns=['beiwe','content','stress','lonely','sad','energy'])
            
                for file in os.listdir(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/'):
                    df = pd.read_csv(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-evening-{file}')
            
                evening_survey_df = evening_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        evening_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        # Weekly Survey Data
        # -------------------
        print('\tProcessing weekly survey data...')
        for participant in os.listdir(parent_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['beiwe','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])
            
                for file in os.listdir(f'{parent_dir}{participant}/survey_answers/{weekly_survey_id}/'):
                    df = pd.read_csv(f'{parent_dir}{participant}/survey_answers/{weekly_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer'],df.loc[10,'answer']]
                    except KeyError:
                        try:
                            participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer']]
                        except:
                            print(f'\t\tProblem with weekly survey {file} for Participant {pid} - Participant most likely did not answer a question')
                            self.move_to_purgatory(f'{parent_dir}{participant}/survey_answers/{weekly_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-weekly-{file}')
            
                weekly_survey_df = weekly_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        weekly_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Never':0,'Almost Never':1,'Sometimes':2,'Fairly Often':3,'Very Often':4,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        # saving
        try:
            morning_survey_df.to_csv(f'../../data/processed/bpeace-morning-survey.csv')
            evening_survey_df.to_csv(f'../../data/processed/bpeace-evening-survey.csv')
            weekly_survey_df.to_csv(f'../../data/processed/bpeace-weekly-survey.csv')
        except:
            return False
 
        return True

    def process_gps(self,resample_rate=1,data_dir='/Volumes/HEF_Dissertation_Research/utx000/bpeace/beiwe/gps/'):
        '''
        Processes the raw gps data into one csv file for each participant and saves into /data/processed/
        
        All GPS data are recorded at 1-second intervals and stored in separate data files for every hour. The
        data are combined into one dataframe per participant, downsampled to 5-minute intervals using the
        mode value for those 5-minutes (after rounding coordinates to five decimal places), and combined into
        a final dataframe that contains all participants' data. 

        Returns True is able to process the data, false otherwise.
        '''
        print('\tProcessing gps data...')

        gps_df = pd.DataFrame()
        for participant in os.listdir(data_dir):
            if len(participant) == 8: # checking to make sure we only look for participant directories
                pid = participant
                print(f'\t\tWorking for Participant: {pid}')
                participant_df = pd.DataFrame() # 
                for file in os.listdir(f'{data_dir}{pid}/gps/'):
                    if file[-1] == 'v': # so we only import cs[v] files
                        try:
                            hourly_df = pd.read_csv(f'{data_dir}{pid}/gps/{file}',usecols=[1,2,3,4,5]) # all columns but UTC
                        except KeyError:
                            print(f'Problem with gps data for {file} for Participant {pid}')
                            self.move_to_purgatory(f'{data_dir}{pid}/gps/{file}',f'../../data/purgatory/{self.study}-{pid}-gps-{file}')
                    
                        if len(hourly_df) > 0: # append to participant df if there were data for that hour
                            participant_df = participant_df.append(hourly_df,ignore_index=True)
                    
                # converting utc to cdt
                participant_df['Time'] = pd.to_datetime(participant_df['UTC time']) - timedelta(hours=6)
                participant_df.set_index('Time',inplace=True)
                # rounding gps and taking the mode for the specified resample rate
                participant_df = round(participant_df,5)
                participant_df = participant_df.resample(f'{resample_rate}T').apply({lambda x: stats.mode(x)[0]})
                # converting values to numeric and removing NaN datapoints
                participant_df.columns = ['UTC Time','Lat','Long','Alt','Accuracy']
                for col in ['Lat','Long','Alt','Accuracy']:
                    participant_df[col] = pd.to_numeric(participant_df[col],errors='coerce')

                participant_df.dropna(inplace=True)
                participant_df['Beiwe'] = pid

                gps_df = gps_df.append(participant_df)

        try:
            gps_df.to_csv(f'../../data/processed/bpeace-gps.csv')
        except Exception as inst:
            print(inst)
            return False

        return True

    def process_accelerometer(self,resample_rate=100,data_dir='/Volumes/HEF_Dissertation_Research/utx000/bpeace/beiwe/accelerometer/'):
        '''
        Processes the raw accelerometer data from each participant into a single csv.

        Accelerometer data are downsampled to 100 millisecond intervals.

        Returns True is able to process the data, false otherwise.
        '''
        print('\tProcessing accelerometer data...')

        accel_df = pd.DataFrame()
        for pt in os.listdir(data_dir):
            if len(pt) == 8:
                print(f'\t\tWorking for Participant: {pt}')
                pt_df = pd.DataFrame()
                for file in os.listdir(f'{data_dir}{pt}/accelerometer/'):
                    file_df = pd.read_csv(f'{data_dir}{pt}/accelerometer/{file}',parse_dates=[1],infer_datetime_format=True)
                    pt_df = pt_df.append(file_df)
                
                pt_df['Time'] = pt_df['UTC time'] - timedelta(hours=6)
                pt_df.set_index(['Time'],inplace=True)
                pt_df.sort_index(inplace=True)
                pt_df = pt_df.resample(f'{resample_rate}ms').mean()
                pt_df.dropna(inplace=True)
                pt_df['Beiwe'] = pt
                
                accel_df = accel_df.append(pt_df)

        try:
            accel_df.to_csv(f'../../data/processed/bpeace-accelerometer.csv')
        except:
            return False

        return True

    def process_noavg_beiwe(self, variable='bluetooth'):
        '''
        Processes beiwe variables that cannot be downsampled. 
        
        Inputs:
        - variable: string of ['bluetooth','wifi','reachability','power_state']

        Returns True is able to process the data, false otherwise.
        '''
        data_dir = f'/Volumes/HEF_Dissertation_Research/utx000/bpeace/beiwe/{variable}/'
        print(f'\tProcessing {variable} data...')

        var_df = pd.DataFrame()
        for pt in os.listdir(data_dir):
            if len(pt) == 8:
                print(f'\t\tWorking for participant {pt}')
                pt_df = pd.DataFrame()
                for file in os.listdir(f'{data_dir}{pt}/{variable}/'):
                    if file[-1] == 'v':
                        temp = pd.read_csv(f'{data_dir}{pt}/{variable}/{file}',parse_dates=[1],infer_datetime_format=True,engine='python')
                        pt_df = pt_df.append(temp)
                    
                pt_df['UTC time'] = pd.to_datetime(pt_df['UTC time'])
                pt_df['Time'] = pt_df['UTC time'] - timedelta(hours=6)
                pt_df.set_index('Time',inplace=True)
                pt_df['Beiwe'] = pt
                
                var_df = var_df.append(pt_df)

        try:
            var_df.to_csv(f'../../data/processed/bpeace-{variable}.csv')
        except Exception as inst:
            print(inst)
            return False

        return True

class utx000():
    '''
    Class used to process utx000 data (Spring 2020 into Summer 2020)
    '''

    def __init__(self,study="utx000",suffix="ux_s20",data_dir="../../data"):
        self.study = study
        self.suffix = suffix
        self.data_dir = data_dir

        self.id_crossover = pd.read_excel(f'{self.data_dir}/raw/utx000/admin/id_crossover.xlsx',sheet_name='all')
        self.beacon_id = pd.read_excel(f'{self.data_dir}/raw/utx000/admin/id_crossover.xlsx',sheet_name='beacon')

        # Beacon Attributes
        self.linear_model = {}
        for file in os.listdir(f"{self.data_dir}/interim/"):
            file_info = file.split("-")
            if len(file_info) == 3:
                if file_info[1] == "linear_model" and file_info[-1] == self.suffix+".csv":
                    try:
                        self.linear_model[file_info[0]] = pd.read_csv(f'{self.data_dir}/interim/{file}',index_col=0)
                    except FileNotFoundError:
                        print(f"Missing offset for {file_info[0]}")
                        self.linear_model[file_info[0]] = pd.DataFrame(data={"beacon":np.arange(1,51),"constant":np.zeros(51),"coefficient":np.ones(51)}).set_index("beacon")

        self.constant_model = {}
        for file in os.listdir(f"{self.data_dir}/interim/"):
            file_info = file.split("-")
            if len(file_info) == 3:
                if file_info[1] == "offset" and file_info[-1] == self.suffix+".csv":
                    try:
                        self.constant_model[file_info[0]] = pd.read_csv(f'{self.data_dir}/interim/{file}',index_col=0)
                    except FileNotFoundError:
                        print(f"Missing offset for {file_info[0]}")
                        self.constant_model[file_info[0]] = pd.DataFrame(data={"beacon":np.arange(1,51),"correction":np.zeros(51)}).set_index("beacon")

        # EMA Attributes
        self.ema_start = datetime(2020,5,13)
        self.ema_end = datetime(2020,9,2)

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def get_ids(self, pid, by_id="beiwe"):
        """
        Gets all ids associated with the given id
        """
        crossover_info = self.id_crossover.loc[self.id_crossover[by_id] == pid].reset_index(drop=True)
        id_list = []
        for id_type in ["redcap","beiwe","beacon","fitbit"]:
            if id_type == by_id:
                id_list.append(pid)
            else:
                id_list.append(crossover_info[id_type][0])
        del crossover_info

        return id_list[0], id_list[1], id_list[2], id_list[3]

    def process_beacon(self, extreme='zscore', resample_rate=2):
        '''
        Combines data from all sensors on all beacons

        Returns True if able to save one dataframe that contains all the data at regular intervals in /data/processed directory
        '''

        averaging_window = int(60 / resample_rate)
        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder=f'{self.data_dir}/raw/utx000/beacon'
        # list of all beacons used in the study
        beacon_list = self.beacon_list = [1,5,6,7,10,11,15,16,17,19,21,22,24,25,26,28,29,30,32,34,36,38,40,44,46] #13,23,41,48
        print('\tProcessing beacon data...\n\t\tReading for beacon:')
        for beacon in beacon_list:

            # correcting the number since the values <10 have leading zero in directory
            number = f'{beacon:02}'
            print(f'\t\t{number}')

            beacon_folder=f'{beacons_folder}/B{number}'
            beacon_df = pd.DataFrame() # dataframe specific to the beacon

            # getting other ids
            beacon_crossover_info = self.id_crossover.loc[self.id_crossover['beacon']==beacon].reset_index(drop=True)
            beiwe = beacon_crossover_info['beiwe'][0]
            fitbit = beacon_crossover_info['fitbit'][0]
            redcap = beacon_crossover_info['redcap'][0]
            del beacon_crossover_info

            def import_and_merge(csv_dir,number,resample_rate=resample_rate):
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
                        #print(f'{inst}; filename: {file}')
                        print(f'Issue encountered while importing {csv_dir}/{file}, skipping...')
                        self.move_to_purgatory(f'{csv_dir}/{file}',f'../../data/purgatory/B{number}-py3-{file}-{self.suffix}')
            
                df = pd.concat(df_list).resample(f'{resample_rate}T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)
                return df

            # Python3 Sensors
            # ---------------
            py3_df = import_and_merge(f'{beacon_folder}/adafruit', number)
            
            # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues)
            if int(number) >= 28:
                print('\t\t\tNo NO2 sensor - removing values')

                py3_df[['CO','T_CO','RH_CO']] = py3_df[['NO2','T_NO2','RH_NO2']]
                py3_df[['NO2','T_NO2','RH_NO2']] = np.nan

            py3_df['CO'] /= 1000 # converting ppb measurements to ppm

            # Python2 Sensors
            # ---------------
            py2_df = import_and_merge(f'{beacon_folder}/sensirion', number)

            # removing data from bad sensors
            if int(number) in [32]:
                print("\t\t\tRemoving PM data")
                for variable in ['PM_C_1','PM_C_2p5','PM_C_10','PM_N_1','PM_N_2p5','PM_N_10']:
                    py2_df[[variable]] = np.nan
                
            # merging py2 and py3 sensor dataframes
            beacon_df = py3_df.merge(right=py2_df,left_index=True,right_index=True,how='outer')
            
            # Adding time for bad RTC
            if beacon == 1:
                beacon_df.index = beacon_df.index + timedelta(hours=140)
            if beacon == 5:
                beacon_df.index = beacon_df.index + timedelta(minutes=1118)
            if beacon == 11:
                beacon_df.index = beacon_df.index + timedelta(days=1827)

            # getting relevant data only
            start_date = self.beacon_id[self.beacon_id['beiwe'] == beiwe]['start_date'].values[0]
            end_date = self.beacon_id[self.beacon_id['beiwe'] == beiwe]['end_date'].values[0]
            beacon_df = beacon_df[start_date:end_date]

            # combing T/RH readings and dropping the bad ones
            beacon_df['temperature_c'] = beacon_df[['T_CO','T_NO2']].mean(axis=1)
            beacon_df['rh'] = beacon_df[['RH_CO','RH_NO2']].mean(axis=1)
            beacon_df.drop(["T_NO2","T_CO","RH_NO2","RH_CO","Temperature [C]","Relative Humidity"],axis=1,inplace=True)

            # dropping unecessary columns
            beacon_df.drop(["Visible","Infrared","eCO2","PM_N_0p5","PM_N_4","PM_C_4"],axis=1,inplace=True)

            # renaming columns
            beacon_df.columns = ["tvoc","lux","no2","co","co2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","temperature_c","rh"]
            beacon_df.index.rename("timestamp",inplace=True)

            # hard-coded values that should be replaced
            for var in beacon_df.columns:
                beacon_df[var].replace(-100,np.nan,inplace=True)
            
            # offsetting measurements with constant (CO and pm2p5) or linear model (others)
            for var in self.linear_model.keys():
                if var in ["co","pm2p5_mass"]:
                    beacon_df[var] -= self.constant_model[var].loc[beacon,"correction"]
                else:
                    beacon_df[var] = beacon_df[var] * self.linear_model[var].loc[beacon,"coefficient"] + self.linear_model[var].loc[beacon,"constant"]

            # variables that should never have anything less than zero (setting to zero)
            for var in ["tvoc","lux","co","no2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass"]:
                beacon_df[var].mask(beacon_df[var] < 0, 0, inplace=True)
            # (setting to nan)
            for var in ["temperature_c","rh"]:
                beacon_df[var].mask(beacon_df[var] < 0, np.nan, inplace=True)
            
            # variables that should never be less than a certain limit
            for var, threshold in zip(["co2","temperature_c","rh"],[200,15,20]):
                beacon_df[var].mask(beacon_df[var] < threshold, np.nan, inplace=True)
            
            # removing extreme values 
            if extreme == 'zscore':
                # zscore greater than 2.5
                for var in beacon_df.columns:
                    beacon_df['z'] = abs(beacon_df[var] - beacon_df[var].mean()) / beacon_df[var].std(ddof=0)
                    beacon_df.loc[beacon_df['z'] > 2.5, var] = np.nan

                beacon_df.drop(['z'],axis=1,inplace=True)
            elif extreme == 'iqr':
                for var in beacon_df.columns:
                    # Computing IQR
                    Q1 = beacon_df[var].quantile(0.25)
                    Q3 = beacon_df[var].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
                    beacon_df[var].mask(beacon_df[var]<Q1-1.5*IQR,np.nan,inplace=True)
                    beacon_df[var].mask(beacon_df[var]>Q3+1.5*IQR,np.nan,inplace=True)
            else:
                print('\t\t\tExtreme values retained')

            # smooting data  
            for var in beacon_df.columns:
                beacon_df[var] = beacon_df[var].rolling(window=averaging_window,center=True,min_periods=int(averaging_window/2)).mean()

            # adding columns for the pt details
            beacon_df['beacon'] = beacon
            beacon_df['beiwe'] = beiwe
            beacon_df['redcap'] = redcap
            
            # adding to overall df
            beacon_data = pd.concat([beacon_data,beacon_df])

        # saving
        try:
            beacon_data.to_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv')
        except:
            return False

        return True

    def process_gps(self, data_dir='/Volumes/HEF_Dissertation_Research/utx000/data/raw/utx000/beiwe/gps/', home=False):
        '''
        Processes the raw gps data into one csv file for each participant and saves into /data/processed/
        
        All GPS data are recorded at 1-second intervals and stored in separate data files for every hour. The
        data are combined into one dataframe per participant, downsampled to 5-minute intervals using the
        mode value for those 5-minutes (after rounding coordinates to five decimal places), and combined into
        a final dataframe that contains all participants' data. 

        Returns True is able to process the data, false otherwise.
        '''
        print('\tProcessing gps data...')

        gps_df = pd.DataFrame()
        for participant in os.listdir(data_dir):
            if len(participant) == 8: # checking to make sure we only look for participant directories
                pid = participant
                print(f'\t\tWorking for Participant: {pid}')
                participant_df = pd.DataFrame() # 
                for file in os.listdir(f'{data_dir}{pid}/gps/'):
                    if file[-1] == 'v': # so we only import cs[v] files
                        try:
                            hourly_df = pd.read_csv(f'{data_dir}{pid}/gps/{file}',usecols=[1,2,3,4,5]) # all columns but UTC
                        except KeyError:
                            print(f'Problem with gps data for {file} for Participant {pid}')
                            self.move_to_purgatory(f'{data_dir}{pid}/gps/{file}',f'../../data/purgatory/{pid}-gps-{file}-{self.suffix}')
                    
                        if len(hourly_df) > 0: # append to participant df if there were data for that hour
                            participant_df = participant_df.append(hourly_df,ignore_index=True)
                    
                # converting utc to cdt
                participant_df['timestamp'] = pd.to_datetime(participant_df['UTC time']) - timedelta(hours=5)
                participant_df.set_index('timestamp',inplace=True)
                # rounding gps and taking the mode for every 5-minutes
                participant_df = round(participant_df,5)
                participant_df = participant_df.resample('5T').apply({lambda x: stats.mode(x)[0]})
                # converting values to numeric and removing NaN datapoints
                participant_df.columns = ['utc','lat','long','altitude','accuracy']
                for col in ['lat','long','altitude','accuracy']:
                    participant_df[col] = pd.to_numeric(participant_df[col],errors='coerce')

                participant_df.dropna(inplace=True)
                if home == True:
                    # getting participant's home coordinates
                    home_coords = self.beacon_id.set_index('beiwe')
                    home_lat = home_coords.loc[pid,'lat']
                    home_long = home_coords.loc[pid,'long']
                    # getting distance
                    R = 6.371*10**6 # radius of the earth in meters
                    participant_df['x_distance'] = abs( R * (participant_df['lat'] - home_lat) * math.pi * math.cos(home_long) / 180) 
                    participant_df['y_distance'] = abs( R * (participant_df['long'] - home_long) * math.pi / 180) 
                    dist = []
                    for i in range(len(participant_df)):
                        dist.append(math.sqrt(math.pow(participant_df.iloc[i,-2],2) + math.pow(participant_df.iloc[i,-1],2)))
                        
                    participant_df['home_distance'] = dist

                participant_df['beiwe'] = pid
                
                gps_df = gps_df.append(participant_df)

        try:
            gps_df.to_csv(f'../../data/processed/beiwe-gps-{self.suffix}.csv')
        except:
            return False

        return True

    def process_weekly_surveys(self, data_dir='../../data/raw/utx000/beiwe/survey_answers/'):
        '''
        Processes raw weekly survey answers. The survey IDs are:
        - eQ2L3J08ChlsdSXXKOoOjyLJ: morning
        - 7TaT8zapOWO0xdtONnsY8CE0: evening
        
        Returns True if able to save two dataframes for morning/evening survey data in /data/processed directory
        '''
        
        def get_survey_id(path_to_dir,survey_type="morning"):
            """
            Gets the survey id for emas
            """
            # defining some variables for ease of understanding - "alt" vars refer to survey IDs from ULG participants
            morning_survey_id = 'eQ2L3J08ChlsdSXXKOoOjyLJ'
            morning_survey_id_alt = 'pJQCg6t6i6RtNqOE8PBvFK8I'
            evening_survey_id = '7TaT8zapOWO0xdtONnsY8CE0'
            evening_survey_id_alt = 'hflVY6iq39s1wd5slGwaCHBY'
            weekly_survey_id = 'lh9veS0aSw2KfrfwSytYjxVr'
            weekly_survey_id_alt = 'MlLhVitzIgRqOTA3jhPWkuc0'

            if survey_type == "morning":
                if os.path.exists(f"{path_to_dir}/{morning_survey_id}/"):
                    morning_ema_id = morning_survey_id
                else:
                    morning_ema_id = morning_survey_id_alt
                return morning_ema_id
            elif survey_type == "evening":
                if os.path.exists(f"{path_to_dir}/{evening_survey_id}/"):
                    evening_ema_id = evening_survey_id
                else:
                    evening_ema_id = evening_survey_id_alt
                return evening_ema_id
            else:
                if os.path.exists(f"{path_to_dir}/{weekly_survey_id}/"):
                    weekly_ema_id = weekly_survey_id
                else:
                    weekly_ema_id = weekly_survey_id_alt
                return weekly_ema_id
        
        # defining the final dataframes to append to
        evening_survey_df = pd.DataFrame()
        morning_survey_df = pd.DataFrame()
        weekly_survey_df = pd.DataFrame()
        
        # Morning Survey Data
        # -------------------
        print('\tProcessing morning survey data...')
        # looping through the participants and then all their data
        for participant in os.listdir(data_dir):
            # making sure we don't read from any hidden directories/files
            if len(participant) == 8:
                pid = participant
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy','TST','SOL','NAW','Restful'])

                ema_id = get_survey_id(f'{data_dir}{participant}/survey_answers',"morning")
                redcap, _, beacon, _ = self.get_ids(pid)

                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{ema_id}/'):
                    # reading raw data
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}')
                    # adding new row
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
                                                                                               df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with morning survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}',f'../../data/purgatory/{pid}-survey-morning-{file}-{self.suffix}')
            
                # adding other ids
                for col, oid in zip(["redcap","beacon"],[redcap,beacon]):
                    participant_df[col] = oid
                # appending participant df to overall df
                morning_survey_df = morning_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
        
        # replacing string values with numeric
        morning_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':np.nan,'NOT_PRESENTED':np.nan,'SKIP QUESTION':np.nan},inplace=True)
        # fixing any string inputs outside the above range
        morning_survey_df['NAW'] = pd.to_numeric(morning_survey_df['NAW'],errors='coerce')
        morning_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy','tst','sol','naw','restful','redcap','beacon']
        morning_survey_df.index.rename("timestamp",inplace=True)
        morning_survey_df = morning_survey_df.sort_index()[self.ema_start:self.ema_end]
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])

                ema_id = get_survey_id(f'{data_dir}{participant}/survey_answers',"evening")
                redcap, _, beacon, _ = self.get_ids(pid)
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{ema_id}/'):
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}',f'../../data/purgatory/{pid}-survey-evening-{file}-{self.suffix}')
            
                # adding other ids
                for col, oid in zip(["redcap","beacon"],[redcap,beacon]):
                    participant_df[col] = oid
                evening_survey_df = evening_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        evening_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':np.nan,'NOT_PRESENTED':np.nan,'SKIP QUESTION':np.nan},inplace=True)
        evening_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy','redcap','beacon']
        evening_survey_df.index.rename("timestamp",inplace=True)
        evening_survey_df = evening_survey_df.sort_index()[self.ema_start:self.ema_end]

        # Weekly Survey Data
        # -------------------
        print('\tProcessing weekly survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])

                ema_id = get_survey_id(f'{data_dir}{participant}/survey_answers',"weekly")
                redcap, _, beacon, _ = self.get_ids(pid)
            
                try:
                    for file in os.listdir(f'{data_dir}{participant}/survey_answers/{ema_id}/'):
                        df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}')
                        try:
                            participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer'],df.loc[10,'answer']]
                        except KeyError:
                            try:
                                participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer']]
                            except:
                                print(f'\t\tProblem with weekly survey {file} for Participant {pid} - Participant most likely did not answer a question')
                                self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{ema_id}/{file}',f'../../data/purgatory/{pid}-survey-weekly-{file}-{self.suffix}')
                    # adding other ids
                    for col, oid in zip(["redcap","beacon"],[redcap,beacon]):
                        participant_df[col] = oid
                    weekly_survey_df = weekly_survey_df.append(participant_df)
                except FileNotFoundError:
                    print(f'\t\tParticipant {pid} does not seem to have submitted any weekly surveys - check directory')
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        weekly_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Never':0,'Almost Never':1,'Sometimes':2,'Fairly Often':3,'Very Often':4,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':np.nan,'NOT_PRESENTED':np.nan,'SKIP QUESTION':np.nan},inplace=True)
        weekly_survey_df.columns = ['beiwe','upset','unable','stressed','confident','your_way','cope','able','top','angered','overcome','redcap','beacon']
        weekly_survey_df.index.rename("timestamp",inplace=True)
        weekly_survey_df = weekly_survey_df.sort_index()[self.ema_start:self.ema_end]

        # saving
        try:
            morning_survey_df.to_csv(f'../../data/processed/beiwe-morning_ema-{self.suffix}.csv')
            evening_survey_df.to_csv(f'../../data/processed/beiwe-evening_ema-{self.suffix}.csv')
            weekly_survey_df.to_csv(f'../../data/processed/beiwe-weekly_ema-{self.suffix}.csv')
        except:
            print("Problem saving Data")
            return False

        return True

    def process_environment_survey(self, data_file='../../data/raw/utx000/surveys/EESurvey_E1_raw.csv'):
        '''
        Processes raw environment survey (first instance) and combines relevant data into processed directory

        Returns True if processed, False otherwise
        '''
        # Helper Functions
        # ----------------
        def check_resonse(row):
            """
            Checks to see if the user responded with anything (given by 1s). Returns 1 indicating the user did NOT respond, else 0.
            """
            s = sum(row)
            if s == 0:
                return 1
            else:
                return 0

        def get_building_type(row):
            for i, building_type in enumerate(["apartment","duplex","house","dormitory","hotel","other"]):
                if row.iloc[i] == 1:
                    return building_type

        def get_allergy(row):
            if row.iloc[0] == "No":
                return 0
            else:
                for i, allergy_intensity in enumerate(["significantly_worse","somewhat_worse","same","somewhat_better","significantly_better"]):
                    if row.iloc[i] == 1:
                        return allergy_intensity
                    
            return 0

        def get_cleaner_location(row):
            d = {"bleach":[],"ammonia":[],"pinesol":[],"vinegar":[],"alcohol":[],"disinfectant_wipes":[],"soap_and_water":[],"floor_cleaners":[]}
            for cleaner in d.keys():
                temp = row[[column for column in row.index if cleaner in column.replace(" ","_").lower()]]
                locs = []

                for i in range(len(temp)):
                    if temp.iloc[i] == 1:
                        locs.append(temp.index[i].split("=")[1][:-1])
                        
                d[cleaner] = (locs)
                
            return d

        print('\tProcessing first environment survey...',end="")
        # data import and cleaning
        ee = pd.read_csv(f"{self.data_dir}/raw/utx000/surveys/EESurvey_E1_labels.csv",parse_dates=["Survey Timestamp"],index_col=0)
        ee.dropna(subset=["Survey Timestamp"], axis="rows",inplace=True)
        ee.drop([column for column in ee.columns if "gender" in column.lower() or "family" in column.lower()],axis="columns",inplace=True)
        ee.replace("Unchecked",0,inplace=True)
        ee.replace("Checked",1,inplace=True)
        # building type
        ee_building_type = ee[[column for column in ee.columns if "What type of building" in column]]
        ee_building_type["building_type"] = ee_building_type.apply(get_building_type, axis="columns")
        ee_building_type.drop([column for column in ee_building_type.columns if "What type of building" in column], axis="columns", inplace=True)
        # window use
        ee_window = ee[[column for column in ee.columns if "your windows" in column.lower()]]
        ee_window.columns = ["change_temperature","fresh_air","both","other_window_use"]
        ee_window["no_window_use"] = ee_window.apply(check_resonse, axis="columns")
        # smell
        ee_smell = ee[[column for column in ee.columns if "smell" in column.lower()]]
        ee_smell.columns = ["stagnant","smelly","earthy","moldy","cooking","fragrant","well_ventilated","obnoxious","other_smell"]
        ee_smell["no_smell"] = ee_smell.apply(check_resonse, axis="columns")
        # allergy
        ee_allergy = ee[[column for column in ee.columns if "allergy" in column.lower()]]
        ee_allergy["allergy_intensity"] = ee_allergy.apply(get_allergy,axis="columns")
        ee_allergy.drop([column for column in ee_allergy.columns if "allergy" in column.lower() and "intensity" not in column.lower()],axis="columns",inplace=True)
        # use of cleaners
        ee_cleaner_use = ee[[column for column in ee.columns if "cleaners been used" in column.lower()]]
        ee_cleaner_use.columns = ["bleach","ammonia","pinesol","vinegar","alcohol","disinfectant_wipes","soap_and_water","floor_cleaners"]
        ee_cleaner_use["no_cleaners"] = ee_cleaner_use.apply(check_resonse, axis="columns")
        # cleaner location
        ee_cleaner_locs = ee[[column for column in ee.columns if "in which rooms" in column.lower()]]
        ee_cleaner_locs["cleaner_locations"] = ee_cleaner_locs.apply(get_cleaner_location,axis="columns")
        ee_cleaner_locs.drop([column for column in ee.columns if "in which rooms" in column.lower()],axis="columns",inplace=True)
        # combining and processing
        data = [ee_building_type,ee_window,ee_smell,ee_allergy,ee_cleaner_use,ee_cleaner_locs]
        for data_df in data:
            data_df.sort_index(inplace=True)

        df = pd.concat(data,axis="columns")
        df.index.rename("redcap",inplace=True)
        # getting other ids
        df = df.merge(right=self.id_crossover,left_index=True,right_on="redcap")
        df.drop(["first","last"],axis="columns",inplace=True)
        # saving
        try:
            df.to_csv(f"{self.data_dir}/processed/redcap-ee_survey-{self.suffix}.csv",index=False)
            print("finished")
        except:
            print("error")
            return False

        return True

    def process_fitbit(self):
        '''
        Processes fitbit data

        Returns True if processed, False otherwise
        '''
        print('\tProcessing Fitbit data...')

        def import_fitbit(filename, data_dir=f"../../data/raw/utx000/fitbit/",verbose=False):
            '''
            Imports the specified file for each participant in the directory

            Inputs:
            - filename: string corresponding to the filename to look for for each participant

            Returns a dataframe with the combined data from all participants
            '''
            print(f"\tReading from file {filename}")
            df = pd.DataFrame()
            for pt in os.listdir(data_dir):
                if pt[0] != ".":
                    if verbose:
                        print(f"\t\tReading for participant {pt}")
                    try:
                        temp = pd.read_csv(f"{data_dir}{pt}/fitbit/fitbit_{filename}.csv", index_col=0, parse_dates=True)
                        if filename[:4] == "intr":
                            temp = process_fitbit_intraday(temp)

                        temp["beiwe"] = pt
                        df = df.append(temp)
                    except FileNotFoundError:
                        print(f"\t\t{pt}: File {filename} not found for participant {pt}")
            df.index.rename("timestamp",inplace=True)       
            return df

        def get_device_df(info_df):
            '''
            Take dictionary-like entries for fitbit info dataframe for each row in a dataframe and makes a new dataframe
            
            Inputs:
            - info_df: the fitbit info dataframe with the dictionary-like entries
            
            Returns a dataframe for the device column
            '''
            
            overall_dict = {}
            for row in range(len(info_df)):
                Dict = ast.literal_eval(info_df['devices'][row])
                if type(Dict) == dict:
                    Dict = Dict
                elif type(Dict) in [tuple,list] and len(Dict) > 1:
                    Dict = Dict[0]
                else:
                    continue

                for key in Dict.keys():
                    overall_dict.setdefault(key, [])
                    overall_dict[key].append(Dict[key])
                # adding in the date of recording
                overall_dict.setdefault('date', [])
                overall_dict['date'].append(info_df.index[row])
                
            df = pd.DataFrame(overall_dict)
            df['timestamp'] = pd.to_datetime(df['date'],errors='coerce')
            df.drop("date",axis=1,inplace=True)
            return df.set_index('timestamp')

        def get_daily_sleep(daily_df,verbose=False):
            '''
            Creates a dataframe with the daily sleep data summarized
            
            Inputs:
            - daily_df: dataframe created from the daily fitbit csv file
            
            Returns a dataframe of the daily sleep data
            '''
            overall_dict = {"dateOfSleep": [],"duration": [],"efficiency":[],"endTime": [],"infoCode": [],"isMainSleep": [],"levels": [],"logId": [],
                    "minutesAfterWakeup": [],"minutesAsleep": [],"minutesAwake": [],"minutesToFallAsleep": [],"startTime": [],"timeInBed": [],
                    "type": [],"date": [],"beiwe": [],"awakeCount": [],"awakeDuration": [],"awakeningsCount": [],"minuteData": [],"restlessCount": [],"restlessDuration": []}
            for row in range(len(daily_df)):
                # in case Fitbit didn't record sleep records for that night - value is NaN
                pt = daily_df['beiwe'][row]
                if verbose:
                    print(f"\t\tWorking for Participant {pt}")
                if type(daily_df['sleep'][row]) == float:
                    continue
                else:
                    Dict = ast.literal_eval(daily_df['sleep'][row])
                    if type(Dict) == dict:
                        Dict = Dict
                    else:
                        Dict = Dict[0]
                    for key in overall_dict.keys():
                        #overall_dict.setdefault(key, [])
                        if key in ["date","beiwe","redcap","beacon"]:
                            pass
                        elif key in Dict.keys():
                            overall_dict[key].append(Dict[key])
                        else:
                            overall_dict[key].append(np.nan)
                    # adding in the date of recording
                    overall_dict.setdefault('date', [])
                    overall_dict['date'].append(daily_df.index[row])
                    # adding beiwe id
                    bid = daily_df['beiwe'][row]
                    overall_dict.setdefault('beiwe', [])
                    overall_dict['beiwe'].append(bid)
                    # adding other ids
                    crossover_info = self.id_crossover.loc[self.id_crossover['beiwe']==bid].reset_index(drop=True)
                    try:
                        bb = crossover_info['beacon'].iloc[0]
                    except IndexError:
                        bb = np.nan
                    try:
                        rid = crossover_info['redcap'].iloc[0]
                    except IndexError:
                        rid = np.nan
                    del crossover_info
                    overall_dict.setdefault('redcap', [])
                    overall_dict['redcap'].append(rid)
                    overall_dict.setdefault('beacon', [])
                    overall_dict['beacon'].append(bb)

            #for key in overall_dict.keys():
                #print(f"{key}: {len(overall_dict[key])}")
            df = pd.DataFrame(overall_dict)
            df['date'] = pd.to_datetime(df['date'],errors='coerce')
            # removing classic sleep stage data
            df = df[df['type'] != 'classic']
            # dropping/renaming columns
            df.drop(["dateOfSleep","infoCode","logId","type","awakeCount","awakeDuration","awakeningsCount","minuteData","restlessCount","restlessDuration"],axis=1,inplace=True)
            df.columns = ["duration_ms","efficiency","end_time","main_sleep","levels","minutes_after_wakeup","minutes_asleep","minutes_awake","minutes_to_sleep","start_time","time_in_bed","date","beiwe","redcap","beacon"]
            # main sleep issues
            df['main_sleep'] = df.apply(
                lambda row: True if np.isnan(row['main_sleep']) and row["minutes_asleep"] > 180 else row['main_sleep'],
                axis=1
            )
            # recalculating se because Fitbit determines it some unknown/incorrect way
            df["efficiency"] = df["minutes_asleep"] / df["time_in_bed"] * 100
            return df

        def get_sleep_stages(daily_sleep):
            '''
            Creates a dataframe for the minute sleep data
            
            Input(s):
            - daily_sleep: dataframe holding the daily sleep data with a column called minuteData
            
            Returns:
            - sleep_stages: a dataframe with sleep stage data for every stage transition
            - summary: a dataframe with the nightly sleep stage information
            '''
            
            data_dict = {'startDate':[],'endDate':[],'dateTime':[],'level':[],'seconds':[],'beiwe':[]}
            summary_dict = {'start_date':[],'end_date':[],'deep_count':[],'deep_minutes':[],'light_count':[],'light_minutes':[],
                            'rem_count':[],'rem_minutes':[],'wake_count':[],'wake_minutes':[],'beiwe':[]}
            for i in range(len(daily_sleep)):
                d0 = pd.to_datetime(daily_sleep.iloc[i,:]["start_time"])
                d1 = pd.to_datetime(daily_sleep.iloc[i,:]["date"])
                try:
                    sleep_dict = daily_sleep.iloc[i,:]["levels"]
                    for key in sleep_dict.keys():
                        if key == 'data': # data without short wake periods
                            temp_data = sleep_dict['data']
                            for temp_data_dict in temp_data:
                                for data_key in temp_data_dict.keys():
                                    data_dict[data_key].append(temp_data_dict[data_key])
                                data_dict['startDate'].append(d0.date())
                                data_dict['endDate'].append(d1.date())
                                data_dict['beiwe'].append(daily_sleep.iloc[i,:]['beiwe'])
                        elif key == 'summary': # nightly summary data - already in dictionary form
                            for summary_key in sleep_dict['summary'].keys():
                                stage_dict = sleep_dict['summary'][summary_key]
                                for stage_key in ['count','minutes']:
                                    summary_dict[f'{summary_key}_{stage_key}'].append(stage_dict[stage_key])
                                
                            summary_dict['start_date'].append(d0.date())
                            summary_dict['end_date'].append(d1.date())
                            summary_dict['beiwe'].append(daily_sleep.iloc[i,:]['beiwe'])
                        else: # shortData or data with short wake periods - don't need
                            pass
                except AttributeError:
                    pass
                
            sleep_stages = pd.DataFrame(data_dict)
            sleep_stages.columns = ['start_date','end_date','time','stage','time_at_stage','beiwe'] # renaming columns
            # adding column for numeric value of sleep stage 
            def numeric_from_str_sleep_stage(row):
                if row['stage'] == 'wake':
                    return 0
                elif row['stage'] == 'light':
                    return 1
                elif row['stage'] == 'deep':
                    return 2
                elif row['stage'] == 'rem':
                    return 3
                else:
                    return np.nan
                
            sleep_stages['value'] = sleep_stages.apply(lambda row: numeric_from_str_sleep_stage(row), axis=1)
            
            summary = pd.DataFrame(summary_dict)
            # getting sol
            sol = sleep_stages.groupby(["beiwe","start_date"]).first().reset_index()
            sol = sol[sol["stage"] == "wake"]
            sol["sol"] = sol["time_at_stage"] / 60
            # getting wol
            wol = sleep_stages.groupby(["beiwe","start_date"]).last().reset_index()
            wol = wol[wol["stage"] == "wake"]
            wol["wol"] = wol["time_at_stage"] / 60
            wol["date"] = pd.to_datetime(wol["end_date"],errors="coerce")
            return sleep_stages, summary, sol[["beiwe","time","sol"]], wol[["beiwe","date","wol"]]

        def process_fitbit_intraday(raw_df,resample_rate=1):
            '''
            Creates dataframe from the intraday measurments

            Inputs:
            - raw_df: dataframe of the raw data from Fitbit
            - resample_rate: integer specifying the minutes to downsample to

            Returns a dataframe indexed by the first column
            '''
            try:
                df = raw_df.resample(f'{resample_rate}T').mean()
            except TypeError:
                print(f"\t\tDataframe is most likely empty ({len(raw_df)})")
                return raw_df
            return df

        daily = import_fitbit("daily_records")
        daily = daily[daily['activities_steps'] > 0 ]
        
        info = import_fitbit("info")

        intra = import_fitbit("intraday_records")
        intra.columns = ["calories","steps","distance","heartrate","beiwe"]

        #device = get_device_df(info)
        print("\t\tProcessing sleep data")
        sleep_daily = get_daily_sleep(daily)
        daily.drop(['activities_heart','sleep'],axis=1,inplace=True)
        daily.columns = ["calories","bmr","steps","distance","sedentary_minutes","lightly_active_minutes","fairly_active_minutes","very_active_minutes","calories_from_activities","bmi","fat","weight","food_calories_logged","water_logged","beiwe"]
        sleep_stages, sleep_stages_summary, sol, wol = get_sleep_stages(sleep_daily)
        sleep_daily.drop(["levels"],axis=1,inplace=True)
        # adding SOL to daily sleep from sleep stages
        sleep_daily = sleep_daily.merge(right=sol,left_on=["beiwe","start_time"],right_on=["beiwe","time"],how="left")
        sleep_daily["sol"].fillna(0,inplace=True)
        # adding WOL to daily sleep from sleep stages
        sleep_daily["date"] = pd.to_datetime(sleep_daily["date"],errors="coerce")
        sleep_daily = sleep_daily.merge(right=wol,on=["beiwe","date"],how="left")
        sleep_daily["wol"].fillna(0,inplace=True)
        sleep_daily.drop("time",axis="columns",inplace=True)
        # saving
        try:
            daily.to_csv(f'../../data/processed/fitbit-daily-{self.suffix}.csv')
            info.to_csv(f'../../data/processed/fitbit-info-{self.suffix}.csv')
            intra.to_csv(f'../../data/processed/fitbit-intraday-{self.suffix}.csv')

            #device.to_csv(f'../../data/processed/bpeace2-fitbit-device.csv')
            sleep_daily.to_csv(f'../../data/processed/fitbit-sleep_daily-{self.suffix}.csv',index=False)
            sleep_stages.to_csv(f'../../data/processed/fitbit-sleep_stages-{self.suffix}.csv',index=False)
            sleep_stages_summary.to_csv(f'../../data/processed/fitbit-sleep_stages_summary-{self.suffix}.csv',index=False)
        except:
            return False

        return True

class wcwh_pilot(wcwh):
    '''
    Class for processing data from the pilot study with the WCWH ambassador families
    '''

    pass

def main():
    '''
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    os.system("clear")
    logger = logging.getLogger(__name__)

    print("Import data from which study?")
    print("\t1. UT1000\n\t2. UT2000\n\t3. UT3000\n\t4. BPEACE\n\t5. UTX000\n\t6. WCWH Pilot")
    ans = int(input("Study Number: "))
    if ans == 1:
        processor = ut1000()
    elif ans == 2:
        processor = ut2000()
    elif ans == 3:
        processor = ut3000()
    elif ans == 4:
        processor = bpeace()
    elif ans == 5:
        processor = utx000()
    elif ans == 6:
        processor = wcwh(study="wcwh_pilot",suffix="wcwh_s20")
    else:
        print("Invalid Choice")
        exit(1)

    print("What type of data?")
    mod = 0 # used to modify answer for old data 
    if ans == 1:
        print("No Options Available")
        exit(0)
    elif ans == 2:
        print('\t1. Beacon')
    elif ans == 3:
        print('\t1. Fitbit Sleep Stages\n\t2. HEH Survey')
        mod = 100
    else:
        print('\t1. Beacon')
        print('\t2. EMAs')
        print('\t3. GPS')
        print('\t4. Accelerometer')
        print('\t5. Bluetooth')
        print('\t6. Power State')
        print('\t7. WiFi')
        print('\t8. Reachability')
        print('\t9. Environment and Experiences Survey')
        print('\t10. Fitbit')
        print('\t11. All')
        
    ans = int(input('Answer: ')) + mod
    all_no = 11 # might change if more data types are added

    # UT3000 Fitbit Sleep Data
    if ans == 101: 
        modality='fitbit'
        var_='sleepStagesDay_merged'
        if processor.process_beiwe_or_fitbit(modality, var_):
            logger.info(f'Data for UT3000 {modality} {var_} processed')
        else:
            logger.error(f'Data for UT3000 {modality} {var_} NOT processed')

    # UT3000 Home Environment Survey
    if ans == 102:
        if processor.process_heh():
            logger.info(f'Data for UT3000 HEH survey processed')
        else:
            logger.error(f'Data for UT3000 HEH survey NOT processed')

    # Beacon Data
    if ans == 1 or ans == all_no:
        if processor.process_beacon():
            logger.info(f'Data for beacons processed')
        else:
            logger.error(f'Data for beacons NOT processed')

    # EMA Data
    if ans == 2 or ans == all_no:
        if processor.process_weekly_surveys():
            logger.info(f'Data for surveys processed')
        else:
            logger.error(f'Data for surveys NOT processed')

    # GPS Data
    if ans == 3 or ans == all_no:
        if processor.process_gps():
            logger.info(f'Data for GPS processed')
        else:
            logger.error(f'Data for GPS NOT processed')

    # Accelerometer Data
    if ans == 4 or ans == all_no:
        if processor.process_accelerometer():
            logger.info(f'Data for accelerometer processed')
        else:
            logger.error(f'Data for accelerometer NOT processed')

    # Bluetooth Data
    if ans == 5 or ans == all_no:
        if processor.process_noavg_beiwe():
            logger.info(f'Data for bluetooth processed')
        else:
            logger.error(f'Data for bluetooth NOT processed')

    # Power state Data
    if ans == 6 or ans == all_no:
        if processor.process_noavg_beiwe(variable='power_state'):
            logger.info(f'Data for power state processed')
        else:
            logger.error(f'Data for power state NOT processed')

    # Wifi Data
    if ans == 7 or ans == all_no:
        if processor.process_noavg_beiwe(variable='wifi'):
            logger.info(f'Data for WiFi processed')
        else:
            logger.error(f'Data for WiFi NOT processed')

    # Reachability Data
    if ans == 8 or ans == all_no:
        if processor.process_noavg_beiwe(variable='reachability'):
            logger.info(f'Data for reachability processed')
        else:
            logger.error(f'Data for reachability NOT processed')

    # EE Survey
    if ans == 9 or ans == all_no:
        if processor.process_environment_survey():
            logger.info(f'Data for environment and experiences survey processed')
        else:
            logger.error(f'Data for environment and experiences survey NOT processed')

    if ans == 10 or ans == all_no:
        if processor.process_fitbit():
            logger.info(f'Data for fitbit processed')
        else:
            logger.error(f'Data for fitbit NOT processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
