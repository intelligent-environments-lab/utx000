# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

def combine_across_studies(dir_string='fitbit',file_string='dailySteps_merged'):
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

    df.to_csv(f'../../data/processed/ut3000_{dir_string}_{file_string}.csv',index=False)
        
    return True

def main():
    '''
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)

    modality='fitbit'
    var_='sleepStagesDay_merged'
    if combine_across_studies(modality, var_):
        logger.info(f'Data for {modality} {var_} processed')
    else:
        logger.error(f'Data for {modality} {var_} NOT processed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
