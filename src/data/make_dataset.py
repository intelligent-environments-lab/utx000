# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

def combine_across_studies(dir_string='fitbit',file_string='dailySteps_merged'):
    '''
    Imports fitbit or beiwe mood data from ut1000 and ut2000 studies and combines into one dataframe
    '''
    df = pd.DataFrame()
    for i in range(2):
        temp = pd.read_csv(f'../../data/raw/ut{i+1}000/{dir_string}/{file_string}.csv')
        temp['study'] = f'ut{i+1}000'
        
        crossover = pd.read_csv(f'../../data/raw/ut{i+1}000/admin/id_crossover.csv')
        if 'Id' in temp.columns:
            temp = pd.merge(left=temp,right=crossover,left_on='Id',right_on='record',how='left')
        elif 'pid' in temp.columns:
            temp = pd.merge(left=temp,right=crossover,left_on='pid',right_on='beiwe',how='left')
        else:
            return False
        
        df = pd.concat([df,temp])

    df.to_csv(f'../../data/processed/ut3000_{dir_string}_{file_string}.csv')
        
    return True

def main():
    '''
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)

    modality='fitbit'
    var_='dailySteps_merged'
    if combine_across_studies(modality, var_):
        logger.info(f'Data for {modality} {var_} processed')
    else:
        logger.error(f'Data for {modality} {var_} NOT processed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='dataset.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()
