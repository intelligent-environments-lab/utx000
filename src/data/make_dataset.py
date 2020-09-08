# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from pathlib import Path

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

    def process_beacon(self,data_dir='/Users/hagenfritz/Projects/utx000/data/raw/ut2000/beacon/'):
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
            beacons.to_csv(f'~/Projects/utx000/data/processed/ut2000-beacon.csv')
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
            temp = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/raw/ut{i+1}000/{dir_string}/{file_string}.csv')
            temp['study'] = f'ut{i+1}000'
            
            # import the id crossover file and attach so we have record, beiwe, and beacon id
            crossover = pd.read_csv(f'/Users/hagenfritz/Projects/utx000/data/raw/ut{i+1}000/admin/id_crossover.csv')
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

        df.to_csv(f'/Users/hagenfritz/Projects/utx000/data/processed/ut3000-{dir_string}-{file_string}.csv',index=False)
            
        return True

    def process_heh(self):
        '''
        Imports and combines heh survey data, cleans up the data, and saves to processed file
        '''

        # Importing data
        heh_1 = pd.read_csv('/Users/hagenfritz/Projects/utx000/data/raw/ut1000/surveys/heh.csv')
        heh_2 = pd.read_csv('/Users/hagenfritz/Projects/utx000/data/raw/ut2000/surveys/heh.csv')
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
        idCross1 = pd.read_csv('~/Projects/utx000/data/raw/ut1000/admin/id_crossover.csv')
        idCross2 = pd.read_csv('~/Projects/utx000/data/raw/ut2000/admin/id_crossover.csv')
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
        heh.to_csv(f'/Users/hagenfritz/Projects/utx000/data/processed/ut3000-heh.csv',index=False) 

        return True

class bpeace2():
    '''
    Class used to process bpeace2 data (Spring 2020 into Summer 2020)
    '''

    def __init__(self):
        self.study = 'bpeace2'

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
        beacon_list = [30,1,21,34,22,28,24,41,26,48,46,25,15,44,23,29,10,16,36,38,40,5,17,6,13,19,32,11,7] # list of all beacons used in the study
        print('Reading for beacon:')
        for beacon in beacon_list:
            print(f'\t{beacon}')
            beacon_df = pd.DataFrame() # dataframe specific to the beacon
            # correcting the number since the values <10 have leading zero in directory
            if beacon < 10:
                number = f'0{beacon}'
            else:
                number = f'{beacon}'

            # Python3 Sensors
            # ---------------
            py3_df = pd.DataFrame() # dataframe for sensors using python3
            for file in os.listdir(f'../../data/raw/bpeace2/beacon/B{number}/adafruit/'):
                try:
                    # reading in raw data (csv for one day at a time) and appending it to the overal dataframe
                    day_df = pd.read_csv(f'../../data/raw/bpeace2/beacon/B{number}/adafruit/{file}',
                                        index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py3_df = pd.concat([py3_df,day_df])
                except Exception as inst:
                    # for whatever reason, some files have header issues - these are moved to purgatory to undergo triage
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'../../data/raw/bpeace2/beacon/B{number}/adafruit/{file}',f'../../data/purgatory/{self.study}-B{number}-py3-{file}')

            py3_df = py3_df.resample('5T').mean() # resampling to 5 minute intervals (raw data is at about 1 min)

            # Python2 Sensors
            # ---------------
            py2_df = pd.DataFrame()
            for file in os.listdir(f'../../data/raw/bpeace2/beacon/B{number}/sensirion/'):
                try:
                    day_df = pd.read_csv(f'../../data/raw/bpeace2/beacon/B{number}/sensirion/{file}',
                                    index_col='Timestamp',parse_dates=True,infer_datetime_format=True)
                    py2_df = pd.concat([py2_df,day_df])
                except Exception as inst:
                    print(f'{inst}; filename: {file}')
                    self.move_to_purgatory(f'../../data/raw/bpeace2/beacon/B{number}/sensirion/{file}',f'../../data/purgatory/{self.study}-B{number}-py2-{file}')
                
            py2_df = py2_df.resample('5T').mean()
                
            # merging python2 and 3 sensor dataframes and adding column for the beacon
            beacon_df = py3_df.merge(right=py2_df,left_index=True,right_index=True,how='outer')
            beacon_df['Beacon'] = beacon
            
            beacon_data = pd.concat([beacon_data,beacon_df])

        # saving
        try:
            beacon_data.to_csv(f'../../data/processed/bpeace2-beacon.csv')
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
        
        # defining the final dataframes to append to
        evening_survey_df = pd.DataFrame()
        morning_survey_df = pd.DataFrame()
        
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
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S')] = [pid,df.loc[4,'answer'],df.loc[5,'answer'],df.loc[6,'answer'],df.loc[7,'answer'],df.loc[8,'answer'],
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
                                'Low energy':0,'Low Energy':0,'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':0,'High Energy':4,
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
                        participant_df.loc[datetime.strptime(file[:-4],'%Y-%m-%d %H_%M_%S')] = [pid,df.loc[0,'answer'],df.loc[1,'answer'],df.loc[2,'answer'],df.loc[3,'answer'],df.loc[4,'answer']]
                    except KeyError:
                        print(f'\t\tProblem with evening survey {file} for Participant {pid} - Participant most likely did not answer a question')
                        self.move_to_purgatory(f'{parent_dir}{participant}/survey_answers/{evening_survey_id}/{file}',f'../../data/purgatory/{self.study}-{pid}-survey-evening-{file}')
            
                evening_survey_df = evening_survey_df.append(participant_df)
            else:
                print(f'\t\tDirectory {participant} is not valid')
                
        evening_survey_df.replace({'Not at all':0,'A little bit':1,'Quite a bit':2,'Very Much':3,
                                'Low energy':0, 'Somewhat low energy':1,'Neutral':2,'Somewhat high energy':3,'High energy':4,
                                'NO_ANSWER_SELECTED':-1,'NOT_PRESENTED':-1,'SKIP QUESTION':-1},inplace=True)

        # saving
        try:
            morning_survey_df.to_csv(f'../../data/processed/bpeace2-morning-survey.csv')
            evening_survey_df.to_csv(f'../../data/processed/bpeace2-evening-survey.csv')
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
    print('\t0. All')
    print('\t1. UT2000 Beacon')
    print('\t2. UT3000 Fitbit Sleep Stages')
    print('\t3. UT3000 HEH Survey')
    print('\t4. BPEACE2 Beacon')
    print('\t5. BPEACE2 Weekly Surveys')
    ans = int(input('Answer: '))
    ut1000_processor = ut1000()
    ut2000_processor = ut2000()
    ut3000_processor = ut3000()
    bpeace2_processor = bpeace2()

    # UT2000 Beacon Data
    if ans == 0 or ans == 1:
        if ut2000_processor.process_beacon():
            logger.info(f'Data for UT2000 beacons processed')
        else:
            logger.error(f'Data for UT2000 beacons NOT processed')

    # UT3000 Fitbit Sleep Data
    if ans == 0 or ans == 2:
        modality='fitbit'
        var_='sleepStagesDay_merged'
        if ut3000_processor.process_beiwe_or_fitbit(modality, var_):
            logger.info(f'Data for UT3000 {modality} {var_} processed')
        else:
            logger.error(f'Data for UT3000 {modality} {var_} NOT processed')

    # UT3000 Home Environment Survey
    if ans == 0 or ans == 3:
        if ut3000_processor.process_heh():
            logger.info(f'Data for UT3000 HEH survey processed')
        else:
            logger.error(f'Data for UT3000 HEH survey NOT processed')

    # BPEACE2 Beacon Data
    if ans == 0 or ans == 4:
        if bpeace2_processor.process_beacon():
            logger.info(f'Data for BPEACE2 beacons processed')
        else:
            logger.error(f'Data for BPEACE2 beacons NOT processed')

    # BPEACE2 Beacon Data
    if ans == 0 or ans == 5:
        if bpeace2_processor.process_weekly_surveys():
            logger.info(f'Data for BPEACE2 morning and evening surveys processed')
        else:
            logger.error(f'Data for BPEACE2 morning and evening surveys NOT processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
