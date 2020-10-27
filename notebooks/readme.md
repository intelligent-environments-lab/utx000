# Some Notes on Notebooks

## Naming Convention

### First Number
The number convention is based on the chronological order of the studies:

| Study Name | Number | Approx. Start Date | Notes |
| --- | --- | --- | --- |
| UT1000 | 1.0 | Fall 2018 | |
| UT2000 | 2.0 | Spring 2019 | |
| BPEACE1 | 3.0 | Spring 2020 | also referred to as utx000 |
| BPEACE2 | 4.0 | Summer 2020 | also referred to as utx000 extension |

Notebooks prefixed with a zero refer to tests, studies, etc. that were not formally recognized most notably calibration events and small-scale tests both limited in participants and time. 

### Second Number
The second number refers to a certain data modality or combination of data modalities. This number convention really only applies to notebooks starting with the BPEACE2 study. The first entry after the initials of the creator refer to the modality used in the notebook. If there is more than one modality included, the modalities should be separated by an underscore. 

### Third Number
The third number refers to different analyses performed for that study and the modality in that study. There are two reserved numbers though:

| Number | Notebook Name | Meaning |
| :-: | :-: | --- |
| 0 | Processing | These notebooks are meant to troubleshoot how to import and process the data from a certain modality. The code from these notebooks is then adapted into the ```make_dataset.py``` source file. |
| 1 | Summary | These notebooks are meant to look a the big picture of available raw data with minimal changes. |
