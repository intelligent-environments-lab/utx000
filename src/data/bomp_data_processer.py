import os
from datetime import datetime
import pandas as pd

def import_raw(start_date,end_date,data_dir="../../../bleed-orange-measure-purple/data/raw/purpleair/",**kwargs):
    """
    Imports the raw data from each device

    Inputs:
        - start_date: datetime corresponding to the file of interest
        - end_date: dateimte corresponding to the file of interest
        - data_dir: location of the raw data files

    Returns a dataframe with data from all devices
    """
    dataset = pd.DataFrame() # dataframe to hold all data
    
    # looping through all files
    for file in os.listdir(data_dir):
        try:
            # data import if the start and end dates correspond to the file name
            s, e = datetime.strptime(file.split(" ")[-2],"%m_%d_%Y"), datetime.strptime(file.split(" ")[-1],"%m_%d_%Y.csv")
            if s == start_date and e == end_date:

                # data import
                temp = pd.read_csv(f"{data_dir}{file}",
                    usecols=["created_at","PM1.0_CF1_ug/m3","PM2.5_ATM_ug/m3","PM10.0_CF1_ug/m3","Temperature_F","Humidity_%"])
                temp.columns = ["timestamp","pm1_mass","pm2p5_mass","pm10_mass","t_f","rh"]
                
                # updating and adding columns
                temp["timestamp"] = pd.to_datetime(temp["timestamp"],infer_datetime_format=True).dt.tz_convert("US/Central")
                temp["device"] = file.split(" ")[0].split("_")[-1]
                
                # adding to overall dataframe
                dataset = dataset.append(temp)

        except IndexError as e:
            print(f"filename: {file} - {e}")

    return dataset