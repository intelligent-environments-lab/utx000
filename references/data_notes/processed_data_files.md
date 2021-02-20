# Processed Data Files Summary
This document is meant to help summarize the data and processes that lead to the creation of the various processed files saved in the `/data/processed/` directory (not available on GitHub). The source code also contains documentation, but this serves as a one-stop-shop that can be easily interpreted. 

_The list of processed files is constantly evolving so the files below do not represent an exhaustive list nor the final list_

## A Note on Naming Convention
The files are named so as to help give some explanation as to what is being presented in them. The general convention is:

> <data_modalities>-<data_represented>-<study_suffix>

Each of the three sections of the file name are separated by hyphens while individual words within a section are separated by underscores. 

### Data Modality
The first section `<data-modalities>` can reference the one or multiple modalities that are present in the file data. For instance, `beacon` would mean that only data from the BEVO Beacons is present while `beacon_fitbit` would indicate the file contains data summarized from both the BEVO Beacon and the Fitbit devices.

### Data Represented
The second section highlights what data from the modality or modalities is present. For instance, `beiwe-morning_ema` would contain data from the morning EMAs distributed via Beiwe. A more complex name like `beacon-fb_and_gps_filtered` would denote beacon data that has been filtered down to only nights with Fitbit and GPS data from participants. However, because the first section does not contain `fitbit` nor `beiwe`, the data in the file is only from the BEVO Beacon. The other tow modalities were simply used to filter the data based on availability of data on those devices.

### Study Suffix
The last section identifies which study the data is taken from. More information on study suffixes can be found on the [Key File](https://github.com/intelligent-environments-lab/utx000/blob/master/references/WCWH%20Study%20Key.xlsx).

## Processed Data Files
Each of the subsections below highlights a certain processed data file and gives a short description of what to expect in the data file and how the data were generated/compiled.

### `beacon-fb_and_gps_filtered`
This file corresponds to beacon data measured during the evenings when participants' Fitbit devices reported sleep data and GPS coordinates from Beiwe confirmed they were at their home location (based on addresses submitted during consent). 

### `beacon-fb_and_gps_filtered_summary`
This file is similar to the one above, but corresponds to _summarized_ beacon data from evenings when participants' Fitbit devices reported sleep data and GPS coordinates from Beiwe confirmed they were at their home location (based on addresses submitted during consent). Rather than provide timeseries data, the columns correspond to summary statistics like mean, median, max, etc. for each pollutant for each night.

### `fitbit-sleep_summary`
This file contains all sleep metrics from Fitbit including daily sleep summaries in addition to sleep stage summaries for all nights, regardless of other variables.

### `beiwe_fitbit-sleep_summary`
This file combines the overlapping sleep metrics from Fitbit and those reported in the morning EMAs. To combine, we merge by participant on the end date of Fitbit sleep and the submission date of the morning EMA. 

### `beiwe_fitbit-beacon_and_gps_filtered_sleep_summary`
This file takes data from the file mentioned directly above and filters it further based on nights when BEVO Beacons reported data for that participant and Beiwe GPS traces confirmed that they were home.

### `fitbit_fitbit-daily_activity_and_sleep`
Oddly named, this file contains activty and sleep data for corresponding days. Activity data from the following sleep event is merged to _all_ sleep summary statistics for that night. The data are merged based on date the activity data are summarized and the sleep end time date minus 1 day. The latter formulation helps control for nights when participants go to sleep after midnight. 

### `all_modalities-fb_and_gps_filtered`
The most complete set of cross-referenced data available in terms of examining effects of all features on Fitbit sleep metrics at once. The features consist of variables from the following sub-modalities. A small discussion on how they were cross-referenced with the Fitbit is given as well.
- **Beacon**: IEQ measurements from all sensors are aggregated into typical statistical summaries (median, mean, etc.) for each night that GPS and Fitbit data are available from the participant. 
- **Self-Report Sleep**: Self-report sleep metrics are merged with Fitbit sleep metrics based on the date of EMA submission with the end date of the Fitbit sleep episode. 
- **Self-Report Mood**: Self-report mood scores prior to the sleep event measured by Fitbit are compared based on timestamps.
- **Fitbit Activity Data**: Activity data measured by Fitbit is compared to the Fitbit sleep event that occurs on that same day or is the next sleep event i.e. the participant falls asleep after midnight. 
