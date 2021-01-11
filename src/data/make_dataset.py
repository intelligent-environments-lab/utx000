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
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy','TST','SOL','NAW','Restful'])
            
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
        # fixing any string inputs outside the above range
        morning_survey_df['NAW'] = pd.to_numeric(morning_survey_df['NAW'],errors='coerce')
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(parent_dir):
            if len(participant) == 8:
                pid = participant
                # pre-allocating dataframe columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])
            
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
                participant_df = pd.DataFrame(columns=['ID','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])
            
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

    def __init__(self):
        self.study = "utx000"
        self.suffix = "ux_s20"
        self.id_crossover = pd.read_excel('../../data/raw/utx000/admin/id_crossover.xlsx',sheet_name='id')
        self.beacon_id = pd.read_excel('../../data/raw/utx000/admin/id_crossover.xlsx',sheet_name='beacon')

        self.co2_offset = pd.read_csv(f'../../data/interim/co2-offset-{self.suffix}.csv',index_col=0)

        self.co_offset = pd.read_csv(f'../../data/interim/co-offset-{self.suffix}.csv',index_col=0)
        self.no2_offset = pd.read_csv(f'../../data/interim/no2-offset-{self.suffix}.csv',index_col=0)

        self.pm1_mass_offset = pd.read_csv(f'../../data/interim/pm1_mass-offset-{self.suffix}.csv',index_col=0)
        self.pm2p5_mass_offset = pd.read_csv(f'../../data/interim/pm2p5_mass-offset-{self.suffix}.csv',index_col=0)
        self.pm10_mass_offset = pd.read_csv(f'../../data/interim/pm10_mass-offset-{self.suffix}.csv',index_col=0)
        self.pm1_number_offset = pd.read_csv(f'../../data/interim/pm1_number-offset-{self.suffix}.csv',index_col=0)
        self.pm2p5_number_offset = pd.read_csv(f'../../data/interim/pm2p5_number-offset-{self.suffix}.csv',index_col=0)
        self.pm10_number_offset = pd.read_csv(f'../../data/interim/pm10_number-offset-{self.suffix}.csv',index_col=0)

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def process_beacon(self, extreme='zscore'):
        '''
        Combines data from all sensors on all beacons

        Returns True if able to save one dataframe that contains all the data at regular intervals in /data/processed directory
        '''

        beacon_data = pd.DataFrame() # dataframe to hold the final set of data
        beacons_folder='../../data/raw/utx000/beacon'
        # list of all beacons used in the study
        beacon_list = self.beacon_list = [1,5,6,10,11,15,16,17,19,21,22,24,25,26,28,29,30,32,34,36,38,40,41,44,46] #13,23,48
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
                        #print(f'{inst}; filename: {file}')
                        print(f'Issue encountered while importing {csv_dir}/{file}, skipping...')
                        self.move_to_purgatory(f'{csv_dir}/{file}',f'../../data/purgatory/B{number}-py3-{file}-{self.suffix}')
            
                df = pd.concat(df_list).resample('5T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)
                return df

            # Python3 Sensors
            # ---------------
            py3_df = import_and_merge(f'{beacon_folder}/adafruit', number)
            
            # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues - see Hagen)
            if int(number) >= 28:
                print('\t\t\tNo NO2 sensor - removing values')

                py3_df[['CO','T_CO','RH_CO']] = py3_df[['NO2','T_NO2','RH_NO2']]
                py3_df[['NO2','T_NO2','RH_NO2']] = np.nan

            # Removing data from bad sensors
            if int(number) in [11,21,24,26]:
                print("\t\t\tRemoving NO2 data")
                py3_df[['NO2']] = np.nan

            py3_df['CO'] /= 1000 # converting ppb measurements to ppm

            # Python2 Sensors
            # ---------------
            py2_df = import_and_merge(f'{beacon_folder}/sensirion', number)

            # removing data from bad sensors
            if int(number) in [32]:
                print("\t\t\tRemoving PM data")
                for variable in ['PM_C_1','PM_C_2p5','PM_C_10','PM_N_1','PM_N_2p5','PM_N_10']:
                    py2_df[[variable]] = np.nan
                
            # merging python2 and 3 sensor dataframes
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
            
            # offsetting measurements
            beacon_df['CO2'] -= self.co2_offset.loc[beacon,'mean']
            beacon_df['CO'] -= self.co_offset.loc[beacon,'mean']
            beacon_df['NO2'] -= self.no2_offset.loc[beacon,'mean']
            beacon_df['PM_C_1'] -= self.pm1_mass_offset.loc[beacon,'mean']
            beacon_df['PM_C_2p5'] -= self.pm2p5_mass_offset.loc[beacon,'mean']
            beacon_df['PM_C_10'] -= self.pm10_mass_offset.loc[beacon,'mean']
            beacon_df['PM_N_1'] -= self.pm1_number_offset.loc[beacon,'mean']
            beacon_df['PM_N_2p5'] -= self.pm2p5_number_offset.loc[beacon,'mean']
            beacon_df['PM_N_10'] -= self.pm10_number_offset.loc[beacon,'mean']
            
            # removing bad values from important variables
            important_vars = ['TVOC','CO2','NO2','CO','PM_C_1','PM_C_2p5','PM_C_10','T_NO2','T_CO','Temperature [C]','Lux','RH_NO2','RH_CO','Relative Humidity']
            
            # variables that should never have anything less than zero
            for var in ['PM_C_1','PM_C_2p5','PM_C_10','T_NO2','T_CO','Temperature [C]','RH_NO2','RH_CO','Relative Humidity']:
                beacon_df[var].mask(beacon_df[var] < 0, np.nan, inplace=True)
            
            # variables that should be corrected to zero if negative
            #for var in ['CO','NO2']:
            #    beacon_df[var].mask(beacon_df[var] < 0, 0, inplace=True)
            
            # variables that should never be less than a certain limit
            for var, threshold in zip(['CO2','Lux'],[100,0]):
                beacon_df[var].mask(beacon_df[var] < threshold, np.nan, inplace=True)
            
            # removing extreme values 
            if extreme == 'zscore':
                # zscore greater than 2.5
                for var in important_vars:
                    beacon_df['z'] = abs(beacon_df[var] - beacon_df[var].mean()) / beacon_df[var].std(ddof=0)
                    beacon_df.loc[beacon_df['z'] > 2.5, var] = np.nan

                beacon_df.drop(['z'],axis=1,inplace=True)
            elif extreme == 'iqr':
                for var in important_vars:
                    # Computing IQR
                    Q1 = beacon_df[var].quantile(0.25)
                    Q3 = beacon_df[var].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
                    beacon_df[var].mask(beacon_df[var]<Q1-1.5*IQR,np.nan,inplace=True)
                    beacon_df[var].mask(beacon_df[var]>Q3+1.5*IQR,np.nan,inplace=True)
            else:
                print('\t\t\tExtreme values retained')

            # dropping NaN values that get in
            beacon_df.dropna(subset=important_vars,how='all',inplace=True)

            # combing T/RH readings and dropping the bad ones
            beacon_df['temperature_c'] = beacon_df[['T_CO','T_NO2']].mean(axis=1)
            beacon_df['rh'] = beacon_df[['RH_CO','RH_NO2']].mean(axis=1)
            beacon_df.drop(["T_NO2","T_CO","RH_NO2","RH_CO","Temperature [C]","Relative Humidity"],axis=1,inplace=True)

            # dropping unecessary columns
            beacon_df.drop(["Visible","Infrared","eCO2","PM_N_0p5","PM_N_4","PM_C_4"],axis=1,inplace=True)

            # renaming columns
            beacon_df.columns = ["tvoc","lux","no2","co","co2","pm1_number","pm2p5_number","pm10_number","pm1_mass","pm2p5_mass","pm10_mass","temperature_c","rh"]
            beacon_df.index.rename("timestamp",inplace=True)

            # adding columns for the pt details
            beacon_df['beacon'] = beacon
            beacon_df['beiwe'] = beiwe
            beacon_df['fitbit'] = fitbit
            beacon_df['redcap'] = redcap
            
            beacon_data = pd.concat([beacon_data,beacon_df])

        # saving
        try:
            beacon_data.to_csv(f'../../data/processed/beacon-{self.suffix}.csv')
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
        # defining some variables for ease of understanding
        morning_survey_id = 'eQ2L3J08ChlsdSXXKOoOjyLJ'
        evening_survey_id = '7TaT8zapOWO0xdtONnsY8CE0'
        weekly_survey_id = 'lh9veS0aSw2KfrfwSytYjxVr'
        
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
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/'):
                    # reading raw data
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/{file}')
                    # adding new row
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
                                                                                               df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with morning survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{morning_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-morning-{file}-{self.suffix}')
            
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
        morning_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy','tst','sol','naw','restful']
        morning_survey_df.index.rename("timestamp",inplace=True)
        
        # Evening Survey Data
        # -------------------
        print('\tProcessing evening survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])
            
                for file in os.listdir(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/'):
                    df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{evening_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-evening-{file}-{self.suffix}')
            
                evening_survey_df = evening_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        evening_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        evening_survey_df.columns = ['beiwe','content','stress','lonely','sad','energy']
        evening_survey_df.index.rename("timestamp",inplace=True)

        # Weekly Survey Data
        # -------------------
        print('\tProcessing weekly survey data...')
        for participant in os.listdir(data_dir):
            if len(participant) == 8:
                pid = participant
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Upset','Unable','Stressed','Confident','Your_Way','Cope','Able','Top','Angered','Overcome'])
            
                try:
                    for file in os.listdir(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/'):
                        df = pd.read_csv(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/{file}')
                        try:
                            participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer'],df.loc[10,'answer']]
                        except KeyError:
                            try:
                                participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=6)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],df.loc[9,'answer']]
                            except:
                                print(f'\t\tProblem with weekly survey {file} for Participant {pid} - Participant most likely did not answer a question')
                                self.move_to_purgatory(f'{data_dir}{participant}/survey_answers/{weekly_survey_id}/{file}',f'../../data/purgatory/{pid}-survey-weekly-{file}-{self.suffix}')
                
                    weekly_survey_df = weekly_survey_df.append(participant_df)
                except FileNotFoundError:
                    print(f'\t\tParticipant {pid} does not seem to have submitted any weekly surveys - check directory')
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        weekly_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Never':0,'Almost Never':1,'Sometimes':2,'Fairly Often':3,'Very Often':4,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)
        weekly_survey_df.columns = ['beiwe','upset','unable','stressed','confident','your_way','cope','able','top','angered','overcome']
        weekly_survey_df.index.rename("timestamp",inplace=True)

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
        print('\tProcessing first environment survey...')

        ee = pd.read_csv(data_file,usecols=[0,2,4,5,6,7,8,9],parse_dates=[1])
        ee.columns = ['redcap','timestamp','apartment','duplex','house','dorm','hotel','other_living']
        ee.dropna(subset=['timestamp'],inplace=True)
        ee.set_index('timestamp',inplace=True)

        # saving
        try:
            ee.to_csv(f'../../data/processed/ee-survey-{self.suffix}.csv')
        except:
            return False

        return True

    def process_fitbit(self):
        '''
        Processes fitbit data

        Returns True if processed, False otherwise
        '''
        print('\tProcessing Fitbit data...')

        def import_fitbit(filename, data_dir=f"../../data/raw/utx000/fitbit/"):
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
                    print(f"\t\tReading for participant {pt}")
                    try:
                        temp = pd.read_csv(f"{data_dir}{pt}/fitbit_{filename}.csv", index_col=0, parse_dates=True)
                        if filename[:4] == "intr":
                            temp = process_fitbit_intraday(temp)

                        temp["beiwe"] = pt
                        df = df.append(temp)
                    except FileNotFoundError:
                        print(f"\t\tFile {filename} not found for participant {pt}")
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

        def get_daily_sleep(daily_df):
            '''
            Creates a dataframe with the daily sleep data summarized
            
            Inputs:
            - daily_df: dataframe created from the daily fitbit csv file
            
            Returns a dataframe of the daily sleep data
            '''
            overall_dict = {}
            for row in range(len(daily_df)):
                # in case Fitbit didn't record sleep records for that night - value is NaN
                pt = daily_df['beiwe'][row]
                # pts with classic sleep data
                if pt in ['awa8uces','ewvz3zm1','pgvvwyvh']:
                    continue
                if type(daily_df['sleep'][row]) == float:
                    continue
                else:
                    Dict = ast.literal_eval(daily_df['sleep'][row])
                    if type(Dict) == dict:
                        Dict = Dict
                    else:
                        Dict = Dict[0]
                    for key in Dict.keys():
                        overall_dict.setdefault(key, [])
                        overall_dict[key].append(Dict[key])
                    # adding in the date of recording
                    overall_dict.setdefault('date', [])
                    overall_dict['date'].append(daily_df.index[row])
                    # adding beiwe id
                    overall_dict.setdefault('beiwe', [])
                    overall_dict['beiwe'].append(daily_df['beiwe'][row])

            df = pd.DataFrame(overall_dict)
            df['date'] = pd.to_datetime(df['date'],errors='coerce')
            # removing classic sleep stage data
            df = df[df['type'] != 'classic']
            # dropping/renaming columns
            df.drop(["dateOfSleep","infoCode","logId","type"],axis=1,inplace=True)
            df.columns = ["duration_ms","efficiency","end_time","main_sleep","levels","minutes_after_wakeup","minutes_asleep","minutes_awake","minutes_to_sleep","start_time","time_in_bed","date","beiwe"]
            df.set_index("date",inplace=True)
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
                    return -1
                
            sleep_stages['value'] = sleep_stages.apply(lambda row: numeric_from_str_sleep_stage(row), axis=1)
            
            summary = pd.DataFrame(summary_dict)
            return sleep_stages, summary

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
        sleep_stages, sleep_stages_summary = get_sleep_stages(sleep_daily)
        sleep_daily.drop(["levels"],axis=1,inplace=True)

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
    print('\t4. All BPEACE Data')
    print('\t5. BPEACE Beacon')
    print('\t6. BPEACE Weekly EMAs')
    print('\t7. BPEACE GPS')
    print('\t8. BPEACE Accelerometer')
    print('\t9. BPEACE Bluetooth')
    print('\t10. BPEACE Power State')
    print('\t11. BPEACE WiFi')
    print('\t12. BPEACE Reachability')
    print('\t13. All UTX000 Data')
    print('\t14. UTX000 Beacon')
    print('\t15. UTX000 Weekly EMAs')
    print('\t16. UTX000 Fitbit')
    print('\t17. UTX000 GPS')
    print('\t18. UTX000 Accelerometer')
    print('\t19. UTX000 Bluetooth')
    print('\t20. UTX000 Power State')
    print('\t21. UTX000 WiFi')
    print('\t22. UTX000 Reachability')
    print('\t23. UTX000 REDCap Environment and Experiences Survey')

    ans = int(input('Answer: '))
    ut1000_processor = ut1000()
    ut2000_processor = ut2000()
    ut3000_processor = ut3000()
    bpeace_processor = bpeace()
    utx000_processor = utx000()

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
    # BPEACE Beacon Data
    if ans == 4 or ans == 5:
        if bpeace_processor.process_beacon():
            logger.info(f'Data for BPEACE beacons processed')
        else:
            logger.error(f'Data for BPEACE beacons NOT processed')

    # BPEACE survey Data
    if ans == 4 or ans == 6:
        if bpeace_processor.process_weekly_surveys():
            logger.info(f'Data for BPEACE surveys processed')
        else:
            logger.error(f'Data for BPEACE surveys NOT processed')

    # BPEACE GPS Data
    if ans == 4 or ans == 7:
        if bpeace_processor.process_gps():
            logger.info(f'Data for BPEACE GPS processed')
        else:
            logger.error(f'Data for BPEACE GPS NOT processed')

    # BPEACE accelerometer Data
    if ans == 4 or ans == 8:
        if bpeace_processor.process_accelerometer():
            logger.info(f'Data for BPEACE accelerometer processed')
        else:
            logger.error(f'Data for BPEACE accelerometer NOT processed')

    # BPEACE bluetooth Data
    if ans == 4 or ans == 9:
        if bpeace_processor.process_noavg_beiwe():
            logger.info(f'Data for BPEACE bluetooth processed')
        else:
            logger.error(f'Data for BPEACE bluetooth NOT processed')

    # BPEACE power state Data
    if ans == 4 or ans == 10:
        if bpeace_processor.process_noavg_beiwe(variable='power_state'):
            logger.info(f'Data for BPEACE power state processed')
        else:
            logger.error(f'Data for BPEACE power state NOT processed')

    # BPEACE Wifi Data
    if ans == 4 or ans == 11:
        if bpeace_processor.process_noavg_beiwe(variable='wifi'):
            logger.info(f'Data for BPEACE WiFi processed')
        else:
            logger.error(f'Data for BPEACE WiFi NOT processed')

    # BPEACE reachability Data
    if ans == 4 or ans == 12:
        if bpeace_processor.process_noavg_beiwe(variable='reachability'):
            logger.info(f'Data for BPEACE reachability processed')
        else:
            logger.error(f'Data for BPEACE reachability NOT processed')

    # UTX000 Beacon Data
    if ans == 13 or ans == 14:
        if utx000_processor.process_beacon():
            logger.info(f'Data for UTX000 beacons processed')
        else:
            logger.error(f'Data for UTX000 beacons NOT processed')

    # UTX000 survey Data
    if ans == 13 or ans == 15:
        if utx000_processor.process_weekly_surveys():
            logger.info(f'Data for UTX000 morning and evening surveys processed')
        else:
            logger.error(f'Data for UTX000 morning and evening surveys NOT processed')

    # UTX000 fitbit
    if ans == 13 or ans == 16:
        if utx000_processor.process_fitbit():
            logger.info(f'Data for UTX000 fitbit processed')
        else:
            logger.error(f'Data for UTX000 fitbit NOT processed')

    # UTX000 gps Data
    if ans == 13 or ans == 17:
        if utx000_processor.process_gps():
            logger.info(f'Data for UTX000 GPS processed')
        else:
            logger.error(f'Data for UTX000 GPS NOT processed')

    # UTX000 EE Survey
    if ans == 13 or ans == 18:
        if utx000_processor.process_environment_survey():
            logger.info(f'Data for UTX000 environment and experiences survey processed')
        else:
            logger.error(f'Data for UTX000 environment and experiences survey NOT processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
