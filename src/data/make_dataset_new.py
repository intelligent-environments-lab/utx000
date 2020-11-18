# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats as stats

from datetime import datetime, timedelta
import math

import os
import logging
from pathlib import Path

import ast

class ut1000():
    '''
    Class dedicated to processing ut1000 data only
    '''

    def __init__(self):
        self.study = 'ut1000'

class ut2000():
    '''
    Class dedicated to processing ut2000 data only
    '''

    def __init__(self):
        self.study = 'ut2000'

    def get_beacon_datetime_index(self,df,resample_rate='10T'):
        '''
        Takes the utc timestamp index, converts it to datetime, sets the index, and resamples
        '''
        dt = []
        for i in range(len(df)):
            if isinstance(df.index[i], str):
                try:
                    ts = int(df.index[i])
                except ValueError:
                    ts = int(df.index[i][:-2])
                dt.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
            else:
                dt.append(datetime.now())

        df['datetime'] = dt
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime',inplace=True)
        df = df.resample('10T').mean()
        
        return df

    def process_beacon(self,data_dir='../../data/raw/ut2000/beacon/'):
        '''
        Combines data from all sensors on all beacons
        '''

        beacons = pd.DataFrame()
        trh = pd.DataFrame()
        measurements = ['pm1.0','pm2.5','pm10','std1.0','std2.5','std10','pc0.3','pc0.5','pc1.0','pc2.5','pc5.0','pc10.0']
        for folder in os.listdir(data_dir):
            beacon_no = folder[-2:]
            if beacon_no in ['07','12','09','03','08','02','01','06','05','10']:
                
                beaconPM = pd.DataFrame()
                for file in os.listdir(f'{data_dir}{folder}/bevo/pms5003/'):
                    if file[-1] == 'v':
                        temp = pd.read_csv(f'{data_dir}{folder}/bevo/pms5003/{file}',names=measurements,
                                          parse_dates=True,infer_datetime_format=True)
                        if len(temp) > 1:
                            beaconPM = pd.concat([beaconPM,temp])
                            
                beaconTVOC = pd.DataFrame()
                for file in os.listdir(f'{data_dir}{folder}/bevo/sgp30/'):
                    if file[-1] == 'v':
                        temp = pd.read_csv(f'{data_dir}{folder}/bevo/sgp30/{file}',
                                        names=['eco2','TVOC'],
                                        parse_dates=True,infer_datetime_format=True)
                        if len(temp) > 1:
                            beaconTVOC = pd.concat([beaconTVOC,temp])
                            
                beaconTRH = pd.DataFrame()
                for file in os.listdir(f'{data_dir}{folder}/bevo/sht31d/'):
                    if file[-1] == 'v':
                        temp = pd.read_csv(f'{data_dir}{folder}/bevo/sht31d/{file}',
                                        names=['RH','TC'],
                                        parse_dates=True,infer_datetime_format=True)
                        if len(temp) > 1:
                            beaconTRH = pd.concat([beaconTRH,temp])
                
                # converting timestamp to datetime, tagging, and combining to overall
                beaconPM = self.get_beacon_datetime_index(beaconPM)
                beaconTVOC = self.get_beacon_datetime_index(beaconTVOC)
                beaconTRH = self.get_beacon_datetime_index(beaconTRH)
                beaconDF = pd.concat([beaconPM,beaconTVOC,beaconTRH],axis=1,join='outer')
                beaconDF['number'] = beacon_no
                beacons = pd.concat([beacons,beaconDF])

        try:
            beacons.to_csv(f'../../data/processed/ut2000-beacon.csv')
        except:
            return False

        return True

class ut3000():
    '''
    Class dedicated to processing ut1000 and ut2000 data together
    '''

    def __init__(self):
        self.study = 'ut3000'

    def process_beiwe_or_fitbit(self,dir_string='fitbit',file_string='dailySteps_merged'):
        '''
        Imports fitbit or beiwe mood data from ut1000 and ut2000, combines them into
        one dataframe, then adds/adjusts columns before writing data to a csv in the
        processed data directory.
        '''
        df = pd.DataFrame()
        for i in range(2):
            # import the file and attach a study tag
            temp = pd.read_csv(f'../../data/raw/ut{i+1}000/{dir_string}/{file_string}.csv')
            temp['study'] = f'ut{i+1}000'
            
            # import the id crossover file and attach so we have record, beiwe, and beacon id
            crossover = pd.read_csv(f'../../data/raw/ut{i+1}000/admin/id_crossover.csv')
            if 'Id' in temp.columns: # fitbit
                temp = pd.merge(left=temp,right=crossover,left_on='Id',right_on='record',how='left')
            elif 'pid' in temp.columns: # beiwe
                temp = pd.merge(left=temp,right=crossover,left_on='pid',right_on='beiwe',how='left')
            else: # neither
                return False
            
            df = pd.concat([df,temp])

        # further processessing based on dir and file strings
        if dir_string == 'fitbit' and file_string == 'sleepStagesDay_merged':
            # removing nights that have no measured sleep
            df = df[df['TotalMinutesLight'] > 0]
            # adding extra sleep metric columns
            df['SleepEfficiency'] = df['TotalMinutesAsleep'] / df['TotalTimeInBed']
            df['TotalMinutesNREM'] = df['TotalMinutesLight'] + df['TotalMinutesDeep'] 
            df['REM2NREM'] = df['TotalMinutesREM'] / df['TotalMinutesNREM']

        df.to_csv(f'../../data/processed/ut3000-{dir_string}-{file_string}.csv',index=False)
            
        return True

    def process_heh(self):
        '''
        Imports and combines heh survey data, cleans up the data, and saves to processed file
        '''

        # Importing data
        heh_1 = pd.read_csv('../../data/raw/ut1000/surveys/heh.csv')
        heh_2 = pd.read_csv('../../data/raw/ut2000/surveys/heh.csv')
        # Dropping all the NaN values from the ut2000 survey
        heh_2.dropna(subset=['livingsit'],inplace=True)
        # Re-mapping choices to numbers - 0 for no, 1 for yes
        heh_1.columns = heh_2.columns
        heh_1.dropna(subset=['livingsit'],inplace=True)
        heh_1['smoke'] = heh_1['smoke'].map({'Yes':1,'No':0})
        heh_1['vape'] = heh_1['vape'].map({'Yes':1,'No':0})
        heh_1['cook_home'] = heh_1['cook_home'].map({'Yes':1,'No':0})
        heh_1['kitchen_exhaust'] = heh_1['kitchen_exhaust'].map({'Yes':1,'No':0})
        heh_1['flu_3w'] = heh_1['flu_3w'].map({'Yes':1,'No':0})
        heh_1['allergies_3w'] = heh_1['allergies_3w'].map({'Yes':1,'No':0})
        heh_1['cold_3w'] = heh_1['cold_3w'].map({'Yes':1,'No':0})
        sameCols = heh_1.columns == heh_2.columns
        for val in sameCols:
            if val == False:
                return False

        # Tagging
        heh_1['study'] = 'ut1000'
        heh_2['study'] = 'ut2000'
        # Adding beiwe and beacon IDs
        idCross1 = pd.read_csv('../../data/raw/ut1000/admin/id_crossover.csv')
        idCross2 = pd.read_csv('../../data/raw/ut2000/admin/id_crossover.csv')
        heh_1 = pd.merge(left=heh_1,left_on='record_id',right=idCross1,right_on='record',how='left')
        heh_2 = pd.merge(left=heh_2,left_on='record_id',right=idCross2,right_on='record',how='left')
        # combining
        heh = pd.concat([heh_1,heh_2], axis=0)
        # Cleaning combined survey
        ## Getting same answers for living situation
        heh['livingsit'] = heh['livingsit'].map({'Apartment':'Apartment','Dormitory':'Dormitory','Stand-alone House':'Stand-alone House',
                                                 3.0:'Apartment',2.0:'Stand-alone House'})
        # Getting just number of roomates
        mates = []
        for i in range(len(heh)):
            r = heh['amt_rmmates'].values[i]
            h = heh['amt_housemates'].values[i]
            if r > 0:
                mates.append(r)
            elif h > 0:
                mates.append(h)
            else:
                mates.append(0)
            
        heh['roommates'] = mates
        heh = heh.drop(['amt_rmmates','amt_housemates'],axis=1)
        # Adding zero where NaN
        heh.fillna(0, inplace=True)
        # saving the file!
        heh.to_csv(f'../../data/processed/ut3000-heh.csv',index=False) 

        return True

class bpeace():
    '''
    Class used to process bpeace data which is made up of two cohorts: Spring and Summer
    '''

    def __init__(self, study):
        self.study = study
        # Defining study-specific variables
        if study == 'bpeace1':
            self.id_crossover = pd.read_excel('../../data/raw/bpeace1/admin/id_crossover.xlsx',sheet_name='id')
            self.beacon_id = pd.read_excel('../../data/raw/bpeace1/admin/id_crossover.xlsx',sheet_name='beacon')
            self.beacon_list = [29,28,27,26,25,24,23,22,20,19,18,17,16,15,14,13,12,11,10,9,7,6,5,4,3,2,1]
            self.utc_difference = 6 #hours
            self.morning_survey_id = 'vBewaVfZ6oWcsiAoPvF6CZi7'
            self.evening_survey_id = 'OymqfwTdyaHFIsJoUNIfPWyG'
            self.weekly_survey_id = 'aMIwBMFUgO8Rtw2ZFjyMTzDn'
        else:
            self.id_crossover = pd.read_excel('../../data/raw/bpeace2/admin/id_crossover.xlsx',sheet_name='id')
            self.beacon_id = pd.read_excel('../../data/raw/bpeace2/admin/id_crossover.xlsx',sheet_name='beacon')
            self.beacon_list = [ 1,  5,  6,  7, 10, 11, 15, 16, 17, 19, 21, 22, 24, 25, 26, 28, 29, 30, 32, 34, 36, 38, 40, 41, 44, 46, 48] #13, 23
            self.utc_difference = 5 #hours
            self.morning_survey_id = 'eQ2L3J08ChlsdSXXKOoOjyLJ'
            self.evening_survey_id = '7TaT8zapOWO0xdtONnsY8CE0'
            self.weekly_survey_id = 'lh9veS0aSw2KfrfwSytYjxVr'

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def process_beacon(self,data_dir='../../data/raw/bpeace1/beacon/',beacon_list=np.arange(0,51,1),resample_rate=5):
        '''
        Combines data from all sensors on all beacons

        Returns True if able to save one dataframe that contains all the data at regular intervals in /data/processed directory
        '''

        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        # list of all beacons used in the study
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
            for file in os.listdir(f'{data_dir}B{number}/adafruit/'):
                try:
                    # reading in raw data (csv for one day at a time) and appending it to the overal dataframe
                    day_df = pd.read_csv(f'{data_dir}B{number}/adafruit/{file}',
                                        index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py3_df = pd.concat([py3_df,day_df])
                except Exception as inst:
                    # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'{data_dir}B{number}/adafruit/{file}',f'../../data/purgatory/{self.study}-B{number}-py3-{file}')

            py3_df = py3_df.resample(f'{resample_rate}T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)
            # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues - see Hagen)
            if number in ['28','29','32','34','36','38','40','46','30','44','48']:
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
            for file in os.listdir(f'{data_dir}B{number}/sensirion/'):
                try:
                    day_df = pd.read_csv(f'{data_dir}B{number}/sensirion/{file}',
                                    index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py2_df = pd.concat([py2_df,day_df])
                except Exception as inst:
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'{data_dir}B{number}/sensirion/{file}',f'../../data/purgatory/{self.study}-B{number}-py2-{file}')
                
            for col in py2_df.columns:
                py2_df[col] = pd.to_numeric(py2_df[col],errors='coerce')

            py2_df = py2_df.resample(f'{resample_rate}T').mean()
                
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
            beacon_data.to_csv(f'../../data/processed/{self.study}-beacon.csv')
        except:
            return False

        return True

    def process_weekly_surveys(self,data_dir='../../data/raw/bpeace1/beiwe/survey_answers/'):
        '''
        Processes raw weekly survey answers. The survey IDs are:
        - eQ2L3J08ChlsdSXXKOoOjyLJ: morning
        - 7TaT8zapOWO0xdtONnsY8CE0: evening
        
        Returns True if able to save two dataframes for morning/evening survey data in /data/processed directory
        '''
        
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
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{self.morning_survey_id}/'):
                    # reading raw data
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{self.morning_survey_id}/{file}')
                    # adding new row
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=self.utc_difference)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
                                                                                               df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with morning survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{self.morning_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-morning-{file}')
            
                # appending participant df to overall df
                morning_survey_df = morning_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
        
        # replacing string values with numeric
        morning_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        # fixing any string inputs outside the above range
        morning_survey_df['NAW'] = pd.to_numeric(morning_survey_df['NAW'],errors='coerce')
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # pre-allocating dataframe columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{self.evening_survey_id}/'):
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{self.evening_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=self.utc_difference)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{self.evening_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-evening-{file}')
            
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
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{self.weekly_survey_id}/'):
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{self.weekly_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=self.utc_difference)] = [pid,df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer'],df.loc[10,'answer']]
                    except KeyError:
                        try:
                            participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=self.utc_difference)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer']]
                        except:
                            print(f'\t\tProblem with weekly survey {file} for Participant {pid} - Participant most likely did not answer a question')
                            self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{self.weekly_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-weekly-{file}')
            
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
            morning_survey_df.to_csv(f'../../data/processed/{self.study}-morning-survey.csv')
            evening_survey_df.to_csv(f'../../data/processed/{self.study}-evening-survey.csv')
            weekly_survey_df.to_csv(f'../../data/processed/{self.study}-weekly-survey.csv')
        except:
            return False
 
        return True

    def process_gps(self,data_dir='/Volumes/HEF_Dissertation_Research/utx000/bpeace1/beiwe/gps/',resample_rate=-1):
        '''
        Processes the raw gps data into one csv file for each participant and saves into /data/processed/
        
        All GPS data are recorded at 1-second intervals and stored in separate data files for every hour. The
        data are combined into one dataframe per participant, downsampled to whatever intervals using the
        mode value for that interaval (after rounding coordinates to five decimal places), and combined into
        a final dataframe that contains all participants' data. 

        Inputs:
        - data_dir: string specifying the location of the data
        - resample_rate: value to downsample to in MINUTES

        Returns:
        - True if able to process the data, false otherwise.
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
                participant_df['Time'] = pd.to_datetime(participant_df['UTC time']) - timedelta(hours=self.utc_difference)
                participant_df.set_index('Time',inplace=True)
                # rounding gps and taking the mode for the specified resample rate
                participant_df = round(participant_df,5)
                if resample_rate != -1:
                    participant_df = participant_df.resample(f'{resample_rate}T').apply({lambda x: stats.mode(x)[0]})
                # converting values to numeric and removing NaN datapoints
                participant_df.columns = ['UTC Time','Lat','Long','Alt','Accuracy']
                for col in ['Lat','Long','Alt','Accuracy']:
                    participant_df[col] = pd.to_numeric(participant_df[col],errors='coerce')

                participant_df.dropna(inplace=True)
                participant_df['Beiwe'] = pid
                
                gps_df = gps_df.append(participant_df)

        try:
            if resample_rate == -1:
                gps_df.to_csv(f'../../data/processed/{self.study}-gps-original.csv')
            else:
                gps_df.to_csv(f'../../data/processed/{self.study}-gps.csv')
        except:
            return False

        return True

    def process_accelerometer(self,data_dir='/Volumes/HEF_Dissertation_Research/utx000/bpeace1/beiwe/accelerometer/',resample_rate=100):
        '''
        Processes the raw accelerometer data from each participant into a single csv.

        Accelerometer data are recorded at odd intervals (not quite sure yet, but are limited to downsampling at an interval
        of 100 millisecond intervals.

        Inputs:
        - data_dir: string specifying the location of the data
        - resample_rate: value to downsample to in MILLISECONDS

        Returns:
        - True is able to process the data, false otherwise.
        '''

        print('\tProcessing accelerometer data...')

        accel_df = pd.DataFrame()
        for pt in os.listdir(data_dir):
            if len(pt) == 8:
                pt_df = pd.DataFrame()
                print(f'\t\tWorking for Participant: {pt}')
                for file in os.listdir(f'{data_dir}{pt}/accelerometer/'):
                    file_df = pd.read_csv(f'{data_dir}{pt}/accelerometer/{file}',parse_dates=[1],infer_datetime_format=True)
                    pt_df = pt_df.append(file_df)
                
                # converting from UTC
                pt_df['Time'] = pt_df['UTC time'] - timedelta(hours=self.utc_difference)
                pt_df.set_index(['Time'],inplace=True)
                pt_df.sort_index(inplace=True)
                # downsampling
                pt_df = pt_df.resample(f'{resample_rate}ms').mean()
                pt_df.dropna(inplace=True)
                pt_df['Beiwe'] = pt
                
                accel_df = accel_df.append(pt_df)

        try:
            accel_df.to_csv(f'../../data/processed/{self.study}-accelerometer.csv')
        except:
            return False

        return True

    def process_noavg_beiwe(self, variable='bluetooth', data_dir=f'/Volumes/HEF_Dissertation_Research/utx000/bpeace1/beiwe/'):
        '''
        Processes beiwe variables that cannot be downsampled. 
        
        Inputs:
        - variable: string of ['bluetooth','wifi','reachability','power_state']

        Returns True is able to process the data, false otherwise.
        '''
        data_dir = f'{data_dir}{variable}/'
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
                pt_df['Time'] = pt_df['UTC time'] - timedelta(hours=self.utc_difference)
                pt_df.set_index('Time',inplace=True)
                pt_df['Beiwe'] = pt
                
                var_df = var_df.append(pt_df)

        try:
            var_df.to_csv(f'../../data/processed/{self.study}-{variable}.csv')
        except Exception as inst:
            print(inst)
            return False

        return True

def main():
    '''
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)
    print('Data Import Table of Contents:')
    print('\t1. UT2000 Beacon')
    print('\t2. UT3000 Fitbit Sleep Stages')
    print('\t3. UT3000 HEH Survey')
    print('\t4. All BPEACE1 Data')
    print('\t5. BPEACE1 Beacon')
    print('\t6. BPEACE1 Weekly EMAs')
    print('\t7. BPEACE1 GPS')
    print('\t8. BPEACE1 Accelerometer')
    print('\t9. BPEACE1 Bluetooth')
    print('\t10. BPEACE1 Power State')
    print('\t11. BPEACE1 WiFi')
    print('\t12. BPEACE1 Reachability')
    print('\t13. All BPEACE2 Data')
    print('\t14. BPEACE2 Beacon')
    print('\t15. BPEACE2 Weekly EMAs')
    print('\t16. BPEACE2 Fitbit')
    print('\t17. BPEACE2 GPS')
    print('\t18. BPEACE2 Accelerometer')
    print('\t19. BPEACE1 Bluetooth')
    print('\t20. BPEACE1 Power State')
    print('\t21. BPEACE1 WiFi')
    print('\t22. BPEACE1 Reachability')
    print('\t23. BPEACE2 REDCap Environment and Experiences Survey')

    ans = int(input('Answer: '))
    ut1000_processor = ut1000()
    ut2000_processor = ut2000()
    ut3000_processor = ut3000()
    bpeace1_processor = bpeace(study='bpeace1')
    bpeace2_processor = bpeace(study='bpeace2')

    # UT2000 Beacon Data
    if ans == 1:
        if ut2000_processor.process_beacon():
            logger.info(f'Data for UT2000 beacons processed')
        else:
            logger.error(f'Data for UT2000 beacons NOT processed')

    # UT3000 Fitbit Sleep Data
    if ans == 2:
        modality='fitbit'
        var_='sleepStagesDay_merged'
        if ut3000_processor.process_beiwe_or_fitbit(modality, var_):
            logger.info(f'Data for UT3000 {modality} {var_} processed')
        else:
            logger.error(f'Data for UT3000 {modality} {var_} NOT processed')

    # UT3000 Home Environment Survey
    if ans == 3:
        if ut3000_processor.process_heh():
            logger.info(f'Data for UT3000 HEH survey processed')
        else:
            logger.error(f'Data for UT3000 HEH survey NOT processed')
    # BPEACE1 Beacon Data
    if ans == 4 or ans == 5:
        if bpeace1_processor.process_beacon():
            logger.info(f'Data for BPEACE1 beacons processed')
        else:
            logger.error(f'Data for BPEACE1 beacons NOT processed')

    # BPEACE1 survey Data
    if ans == 4 or ans == 6:
        if bpeace1_processor.process_weekly_surveys():
            logger.info(f'Data for BPEACE1 surveys processed')
        else:
            logger.error(f'Data for BPEACE1 surveys NOT processed')

    # BPEACE1 GPS Data
    if ans == 4 or ans == 7:
        if bpeace1_processor.process_gps():
            logger.info(f'Data for BPEACE1 GPS processed')
        else:
            logger.error(f'Data for BPEACE1 GPS NOT processed')

    # BPEACE1 accelerometer Data
    if ans == 4 or ans == 8:
        if bpeace1_processor.process_accelerometer():
            logger.info(f'Data for BPEACE1 accelerometer processed')
        else:
            logger.error(f'Data for BPEACE1 accelerometer NOT processed')

    # BPEACE1 bluetooth Data
    if ans == 4 or ans == 9:
        if bpeace1_processor.process_noavg_beiwe():
            logger.info(f'Data for BPEACE1 bluetooth processed')
        else:
            logger.error(f'Data for BPEACE1 bluetooth NOT processed')

    # BPEACE1 power state Data
    if ans == 4 or ans == 10:
        if bpeace1_processor.process_noavg_beiwe(variable='power_state'):
            logger.info(f'Data for BPEACE1 power state processed')
        else:
            logger.error(f'Data for BPEACE1 power state NOT processed')

    # BPEACE1 Wifi Data
    if ans == 4 or ans == 11:
        if bpeace1_processor.process_noavg_beiwe(variable='wifi'):
            logger.info(f'Data for BPEACE1 WiFi processed')
        else:
            logger.error(f'Data for BPEACE1 WiFi NOT processed')

    # BPEACE1 reachability Data
    if ans == 4 or ans == 12:
        if bpeace1_processor.process_noavg_beiwe(variable='reachability'):
            logger.info(f'Data for BPEACE1 reachability processed')
        else:
            logger.error(f'Data for BPEACE1 reachability NOT processed')

    # BPEACE2 Beacon Data
    if ans == 13 or ans == 14:
        if bpeace2_processor.process_beacon():
            logger.info(f'Data for BPEACE2 beacons processed')
        else:
            logger.error(f'Data for BPEACE2 beacons NOT processed')

    # BPEACE2 survey Data
    if ans == 13 or ans == 15:
        if bpeace2_processor.process_weekly_surveys():
            logger.info(f'Data for BPEACE2 morning and evening surveys processed')
        else:
            logger.error(f'Data for BPEACE2 morning and evening surveys NOT processed')

    # BPEACE2 fitbit
    if ans == 13 or ans == 16:
        if bpeace2_processor.process_fitbit():
            logger.info(f'Data for BPEACE2 fitbit processed')
        else:
            logger.error(f'Data for BPEACE2 fitbit NOT processed')

    # BPEACE2 gps Data
    if ans == 13 or ans == 17:
        if bpeace2_processor.process_gps(data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/gps/'):
            logger.info(f'Data for BPEACE2 GPS processed')
        else:
            logger.error(f'Data for BPEACE2 GPS NOT processed')

    # BPEACE2 accelerometer Data
    if ans == 13 or ans == 18:
        if bpeace2_processor.process_accelerometer(data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/accelerometer/'):
            logger.info(f'Data for BPEACE2 accelerometer processed')
        else:
            logger.error(f'Data for BPEACE2 accelerometer NOT processed')

    # BPEACE2 bluetooth Data
    if ans == 13 or ans == 19:
        if bpeace2_processor.process_noavg_beiwe(variable='bluetooth',data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/'):
            logger.info(f'Data for BPEACE2 bluetooth processed')
        else:
            logger.error(f'Data for BPEACE2 bluetooth NOT processed')

    # BPEACE2 power state Data
    if ans == 13 or ans == 20:
        if bpeace2_processor.process_noavg_beiwe(variable='power_state',data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/'):
            logger.info(f'Data for BPEACE2 power state processed')
        else:
            logger.error(f'Data for BPEACE2 power state NOT processed')

    # BPEACE2 wifi Data
    if ans == 13 or ans == 21:
        if bpeace2_processor.process_noavg_beiwe(variable='wifi',data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/'):
            logger.info(f'Data for BPEACE2 wifi processed')
        else:
            logger.error(f'Data for BPEACE2 wifi NOT processed')

    # BPEACE2 reachability Data
    if ans == 13 or ans == 22:
        if bpeace2_processor.process_noavg_beiwe(variable='reachability',data_dir='/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/'):
            logger.info(f'Data for BPEACE2 reachability processed')
        else:
            logger.error(f'Data for BPEACE2 reachability NOT processed')

    # BPEACE2 EE Survey
    if ans == 13 or ans == 23:
        if bpeace2_processor.process_environment_survey():
            logger.info(f'Data for BPEACE2 environment and experiences survey processed')
        else:
            logger.error(f'Data for BPEACE2 environment and experiences survey NOT processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
