# Ventilation 
# -----------
# Author: Hagen
# Date: 03/29/21
# Description: 
# Data Files Needed:
#   - beacon
#   - beacon-fb_and_gps_filtered
#   - 

import logging

import pandas as pd

class calculate():

    def __init__(self, study="utx000", study_suffix="ux_s20", data_dir="../../data"):
        """ """
        self.suffix = study_suffix
        self.data_dir = data_dir
        self.study = study

        # beacon data
        self.beacon_all = pd.read_csv(f'{self.data_dir}/processed/beacon-{self.suffix}.csv',index_col="timestamp",parse_dates=True,infer_datetime_format=True)
        self.beacon_nightly = pd.read_csv(f'{self.data_dir}/processed/beacon-fb_and_gps_filtered-{self.suffix}.csv',
            index_col="timestamp",parse_dates=["timestamp","start_time","end_time"],infer_datetime_format=True)

        # participant information
        pt_names = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='all')
        pt_names = pt_names[["beiwe","first","last"]]
        pt_ids = pd.read_excel(f'{self.data_dir}/raw/{self.study}/admin/id_crossover.xlsx',sheet_name='beacon')
        pt_ids = pt_ids[['redcap','beiwe','beacon','lat','long','volume','roommates']] # keep their address locations
        self.pt_info = pt_ids.merge(right=pt_names,on='beiwe')

#class steady_state(calculate):

#class decay(calculate):

def main():
    ventilation_estimate = calculate()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='ventilation.log', filemode='w', level=logging.DEBUG, format=log_fmt)

    main()