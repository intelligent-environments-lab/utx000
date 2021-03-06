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

class bpeace2():
    '''
    Class used to process bpeace2 data (Spring 2020 into Summer 2020)
    '''

    def __init__(self):
        self.study = 'bpeace2'
        self.id_crossover = pd.read_excel('../../data/raw/bpeace2/admin/id_crossover.xlsx',sheet_name='id')
        self.beacon_id = pd.read_excel('../../data/raw/bpeace2/admin/id_crossover.xlsx',sheet_name='beacon')

    def move_to_purgatory(self,path_to_file,path_to_destination):
        '''
        Moves problematic file to the purgatory data directory

        Returns void
        '''
        print('\t\tMoving to purgatory...')
        os.replace(path_to_file, path_to_destination)

    def process_beacon(self, remove_extreme=True):
        '''
        Combines data from all sensors on all beacons

        Returns True if able to save one dataframe that contains all the data at regular intervals in /data/processed directory
        '''
        # list of all beacons used in the study
        beacon_list = [1,5,6,7,10,11,15,16,17,19,21,22,23,24,25,26,28,29,30,32,34,36,38,40,41,44,46,48]
        beacons_folder='../../data/raw/bpeace2/beacon'

        beacon_data_list = [] # list where each item will be a dataframe for a beacon
        print('\nProcessing beacon data...')
        for beacon in beacon_list:
            # correcting the number since the values <10 have leading zero in directory
            number = f'{beacon:02}'
            print(f'\t{number}')

            beacon_folder=f'{beacons_folder}/B{number}'
            beacon_df = pd.DataFrame() # dataframe specific to the beacon
             
            # getting other ids
            beacon_crossover_info = self.id_crossover.loc[self.id_crossover['Beacon']==beacon].reset_index(drop=True)
            beiwe = beacon_crossover_info['Beiwe'][0]
            fitbit = beacon_crossover_info['Fitbit'][0]
            redcap = beacon_crossover_info['REDCap'][0]
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
                        self.move_to_purgatory(f'{csv_dir}/{file}',f'../../data/purgatory/{self.study}-B{number}-py3-{file}')
            
                df = pd.concat(df_list).resample('5T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)
                return df
            
            # Python3 Sensors
            # ---------------
            # dataframe for sensors using python3
            py3_df = import_and_merge(f'{beacon_folder}/adafruit', number)
            
            # Changing NO2 readings on beacons without NO2 readings to CO (wiring issues - see Hagen)
            if number in ['28','29','32','34','36','38','40','46','30','44']:
                print('\t\t\tNo NO2 sensor - removing values')
                
                py3_df[['CO','T_CO','RH_CO']] = py3_df[['NO2','T_NO2','RH_NO2']]
                py3_df[['NO2','T_NO2','RH_NO2']] = np.nan

            py3_df['CO'] /= 1000 # converting ppb measurements to ppm

            # Python2 Sensors
            # ---------------
            # dataframe for sensors using python3
            py2_df = import_and_merge(f'{beacon_folder}/sensirion', number)
            
            # merging python2 and 3 sensor dataframes
            beacon_df = py3_df.merge(right=py2_df,left_index=True,right_index=True,how='outer')
            
            # getting relevant data only
            beacon_info = self.beacon_id[self.beacon_id['Beiwe'] == beiwe]
            start_date = beacon_info['start_date'].values[0]
            end_date = beacon_info['end_date'].values[0]
            beacon_df = beacon_df[start_date:end_date]
            del beacon_info
            
            # removing bad values from important variables
            important_vars = ['TVOC','CO2','NO2','CO','PM_C_2p5','PM_C_10','T_NO2','T_CO','Temperature [C]','Lux','RH_NO2','RH_CO','Relative Humidity']
            # variables that should never have anything less than zero
            for var in ['CO2','T_NO2','T_CO','Temperature [C]','RH_NO2','RH_CO','Relative Humidity']:
                beacon_df[var].mask(beacon_df[var] < 0, np.nan, inplace=True)
            # variables that should never be less than a certain limit
            for var, threshold in zip(['CO2','Lux'],[100,-1]):
                beacon_df[var].mask(beacon_df[var] < threshold, np.nan, inplace=True)
            
            if remove_extreme:
                # removing extreme values (zscore greater than 2.5)
                # for var in important_vars:
                #     beacon_df['z'] = abs(beacon_df[var] - beacon_df[var].mean()) / beacon_df[var].std(ddof=0)
                #     beacon_df.loc[beacon_df['z'] > 2.5, var] = np.nan
                # beacon_df.drop(columns='z',inplace=True)
                
                #IQR method
                for var in important_vars:
                    # Computing IQR
                    Q1 = beacon_df[var].quantile(0.25)
                    Q3 = beacon_df[var].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
                    beacon_df[var].mask(beacon_df[var]<Q1-1.5*IQR,np.nan,inplace=True)
                    beacon_df[var].mask(beacon_df[var]>Q3+1.5*IQR,np.nan,inplace=True)
                
            
            # adding columns for the pt details
            beacon_df['Beacon'] = beacon
            beacon_df['Beiwe'] = beiwe
            beacon_df['Fitbit'] = fitbit
            beacon_df['REDCap'] = redcap
            
            beacon_data_list.append(beacon_df)
            
        beacon_data = pd.concat(beacon_data_list)
        # saving
        try:
            filename = f'../../data/processed/bpeace2-beacon-IQR.parquet'
            if not remove_extreme:
                filename = f'../../data/processed/bpeace2-beacon-with-extreme.parquet'
            beacon_data.to_parquet(filename, compression='brotli')
        except:
            return False

        return True

    def process_gps(self, data_dir = '/Volumes/HEF_Dissertation_Research/utx000/extension/data/beiwe/gps/'):
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
                participant_df['Time'] = pd.to_datetime(participant_df['UTC time']) - timedelta(hours=5)
                participant_df.drop(['UTC time'],axis=1,inplace=True)
                participant_df.set_index('Time',inplace=True)
                # rounding gps and taking the mode for every 5-minutes
                participant_df = round(participant_df,5)
                participant_df = participant_df.resample('5T').apply({lambda x: stats.mode(x)[0]})
                # converting values to numeric and removing NaN datapoints
                participant_df.columns = ['Lat','Long','Alt','Accuracy']
                for col in ['Lat','Long','Alt','Accuracy']:
                    participant_df[col] = pd.to_numeric(participant_df[col],errors='coerce')

                participant_df.dropna(inplace=True)
                # getting participant's home coordinates
                home_coords = self.beacon_id.set_index('Beiwe')
                home_lat = home_coords.loc[pid,'Lat']
                home_long = home_coords.loc[pid,'Long']
                # getting distance
                R = 6.371*10**6 # radius of the earth in meters
                participant_df['X_Distance'] = abs( R * (participant_df['Lat'] - home_lat) * math.pi * math.cos(home_long) / 180) 
                participant_df['Y_Distance'] = abs( R * (participant_df['Long'] - home_long) * math.pi / 180) 
                dist = []
                for i in range(len(participant_df)):
                    dist.append(math.sqrt(math.pow(participant_df.iloc[i,-2],2) + math.pow(participant_df.iloc[i,-1],2)))
                    
                participant_df['Distance_Home'] = dist
                participant_df['Beiwe'] = pid
                
                gps_df = gps_df.append(participant_df)

        try:
            gps_df.to_csv(f'../../data/processed/bpeace2-gps.csv')
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
        parent_dir = '../../data/raw/bpeace2/beiwe/survey_answers/'
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
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
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
                # less columns
                participant_df = pd.DataFrame(columns=['ID','Content','Stress','Lonely','Sad','Energy'])
            
                for file in os.listdir(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/'):
                    df = pd.read_csv(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/{file}')
                    try:
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S') - timedelta(hours=5)] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
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
            
                try:
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
                except FileNotFoundError:
                    print(f'\t\tParticipant {pid} does not seem to have submitted any weekly surveys - check directory')
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        weekly_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Never':0,'Almost Never':1,'Sometimes':2,'Fairly Often':3,'Very Often':4,
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,'High Energy':4,
                                'Not at all restful':0,'Slightly restful':1,'Somewhat restful':2,'Very restful':3,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)

        # saving
        try:
            morning_survey_df.to_csv(f'../../data/processed/bpeace2-morning-survey.csv')
            evening_survey_df.to_csv(f'../../data/processed/bpeace2-evening-survey.csv')
            weekly_survey_df.to_csv(f'../../data/processed/bpeace2-weekly-survey.csv')
        except:
            return False

        return True

    def process_environment_survey(self, data_file='../../data/raw/bpeace2/surveys/EESurvey_E1_raw.csv'):
        '''
        Processes raw environment survey (first instance) and combines relevant data into processed directory

        Returns True if processed, False otherwise
        '''
        print('\tProcessing first environment survey...')

        ee = pd.read_csv(data_file,usecols=[0,2,4,5,6,7,8,9],parse_dates=[1])
        ee.columns = ['REDCap','Timestamp','Apartment','Duplex','House','Dorm','Hotel','Other']
        ee.dropna(subset=['Timestamp'],inplace=True)
        ee.set_index('Timestamp',inplace=True)

        # saving
        try:
            ee.to_csv(f'../../data/processed/bpeace2-ee-survey.csv')
        except:
            return False

        return True

    def process_fitbit(self):
        '''
        Processes fitbit data

        Returns True if processed, False otherwise
        '''
        print('\tProcessing Fitbit data...')

        def import_fitbit(filename, pt_dir=f"../../data/raw/bpeace2/fitbit/"):
            '''
            Imports the specified file for each participant in the directory

            Inputs:
            - filename: string corresponding to the filename to look for for each participant

            Returns a dataframe with the combined data from all participants
            '''
            print(f"\tReading from file {filename}")
            df = pd.DataFrame()
            for pt in os.listdir(pt_dir):
                if pt[0] != ".":
                    print(f"\t\tReading for participant {pt}")
                    try:
                        temp = pd.read_csv(f"{pt_dir}{pt}/fitbit_{filename}.csv", index_col=0, parse_dates=True)
                        if filename[:4] == "intr":
                            temp = process_fitbit_intraday(temp)

                        temp["beiwe"] = pt
                        df = df.append(temp)
                    except FileNotFoundError:
                        print(f"\t\tFile {filename} not found for participant {pt}")
                        
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
            df['date'] = pd.to_datetime(df['date'],errors='coerce')
            return df.set_index('date')

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
            return df.set_index('date')

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
            summary_dict = {'startDate':[],'endDate':[],'deep_count':[],'deep_minutes':[],'light_count':[],'light_minutes':[],
                            'rem_count':[],'rem_minutes':[],'wake_count':[],'wake_minutes':[],'beiwe':[]}
            for row in range(len(daily_sleep)):
                d0 = pd.to_datetime(daily_sleep['startTime'][row])
                d1 = pd.to_datetime(daily_sleep['dateOfSleep'][row])
                sleep_dict = daily_sleep['levels'][row]
                for key in sleep_dict.keys():
                    if key == 'data': # data without short wake periods
                        temp_data = sleep_dict['data']
                        for temp_data_dict in temp_data:
                            for data_key in temp_data_dict.keys():
                                data_dict[data_key].append(temp_data_dict[data_key])
                            data_dict['startDate'].append(d0.date())
                            data_dict['endDate'].append(d1.date())
                            data_dict['beiwe'].append(daily_sleep['beiwe'][row])
                    elif key == 'summary': # nightly summary data - already in dictionary form
                        for summary_key in sleep_dict['summary'].keys():
                            stage_dict = sleep_dict['summary'][summary_key]
                            for stage_key in ['count','minutes']:
                                summary_dict[f'{summary_key}_{stage_key}'].append(stage_dict[stage_key])
                            
                        summary_dict['startDate'].append(d0.date())
                        summary_dict['endDate'].append(d1.date())
                        summary_dict['beiwe'].append(daily_sleep['beiwe'][row])
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
            
            '''
            try:
                df = raw_df.resample(f'{resample_rate}T').mean()
            except TypeError:
                print(f"\t\tDataframe is most likely empty ({len(raw_df)})")
                return raw_df
            return df

        daily = import_fitbit("daily_records")
        info = import_fitbit("info")
        intra = import_fitbit("intraday_records")

        #device = get_device_df(info)
        print("\t\tProcessing sleep data")
        sleep_daily = get_daily_sleep(daily)
        sleep_stages, sleep_stages_summary = get_sleep_stages(sleep_daily)

        # some cleaning
        daily.drop(['activities_heart','sleep'],axis=1,inplace=True)
        daily = daily[daily['activities_steps'] > 0 ]
        sleep_daily.drop(['levels','type'],axis=1,inplace=True)

        # saving
        try:
            daily.to_csv(f'../../data/processed/bpeace2-fitbit-daily.csv')
            info.to_csv(f'../../data/processed/bpeace2-fitbit-info.csv')
            intra.to_csv(f'../../data/processed/bpeace2-fitbit-intraday.csv')

            #device.to_csv(f'../../data/processed/bpeace2-fitbit-device.csv')
            sleep_daily.to_csv(f'../../data/processed/bpeace2-fitbit-sleep-daily.csv')
            sleep_stages.to_csv(f'../../data/processed/bpeace2-fitbit-sleep-stages.csv')
            sleep_stages_summary.to_csv(f'../../data/processed/bpeace2-fitbit-sleep-stages-summary.csv')
        except:
            return False

        return True
    
def main():
    '''
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)
    
    print('\t8. All BPEACE2 Data')
    print('\t9. BPEACE2 Beacon')
    print('\t10. BPEACE2 Weekly EMAs')
    print('\t11. BPEACE2 Fitbit')
    # print('\t12. BPEACE2 GPS')
    # print('\t13. BPEACE2 REDCap Environment and Experiences Survey')

    ans = int(input('Answer: '))

    bpeace2_processor = bpeace2()

    # BPEACE2 Beacon Data
    if ans == 8 or ans == 9:
        if bpeace2_processor.process_beacon():
            logger.info(f'Data for BPEACE2 beacons processed')
        else:
            logger.error(f'Data for BPEACE2 beacons NOT processed')

    # # BPEACE2 survey Data
    # if ans == 8 or ans == 10:
    #     if bpeace2_processor.process_weekly_surveys():
    #         logger.info(f'Data for BPEACE2 morning and evening surveys processed')
    #     else:
    #         logger.error(f'Data for BPEACE2 morning and evening surveys NOT processed')

    # # BPEACE2 fitbit
    # if ans == 8 or ans == 11:
    #     if bpeace2_processor.process_fitbit():
    #         logger.info(f'Data for BPEACE2 fitbit processed')
    #     else:
    #         logger.error(f'Data for BPEACE2 fitbit NOT processed')

    # BPEACE2 gps Data
    # if ans == 8 or ans == 12:
    #     if bpeace2_processor.process_gps():
    #         logger.info(f'Data for BPEACE2 GPS processed')
    #     else:
    #         logger.error(f'Data for BPEACE2 GPS NOT processed')

    # BPEACE2 EE Survey
    # if ans == 8 or ans == 13:
    #     if bpeace2_processor.process_environment_survey():
    #         logger.info(f'Data for BPEACE2 environment and experiences survey processed')
    #     else:
    #         logger.error(f'Data for BPEACE2 environment and experiences survey NOT processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
