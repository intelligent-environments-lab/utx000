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
        pass

class ut2000():
    '''
    Class dedicated to processing ut2000 data only
    '''

    def __init__(self):
        pass

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
        pass

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

        # Tagging and combining
        heh_1['study'] = 'ut1000'
        heh_2['study'] = 'ut2000'
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
    ans = int(input('Answer: '))
    ut1000_processor = ut1000()
    ut2000_processor = ut2000()
    ut3000_processor = ut3000()

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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
