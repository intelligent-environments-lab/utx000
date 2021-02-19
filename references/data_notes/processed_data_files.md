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
