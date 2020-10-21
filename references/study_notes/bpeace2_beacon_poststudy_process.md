# BPEACE2 (UTx000 Extension) Beacon Post-Study Process
This document walks the user through the processes necessary to conduct after receiving the beacons from the BPEACE2, also called UTx000 Extension, study.

## Beacon Data Dump

1. Connect beacon to monitor
2. Change wifi to home network
3. Check VPN IP4 address
  - update if necessary on local machine in /etc/localhosts
4. Download data from terminal using ```python3 ~/Projects/bevo_iaq/dldata.py```
5. Remove data from terminal using ```python3 ~/Projects/bevo_iaq/rmdata.py```
6. Change wifi to utexas network by commenting out home network

## Beacon Post-Calibration

1. Move beacons to UTest House
2. Plug in and check connectivity to utexas wifi network
3. Allow to run against reference machine for at least 24 hours
  - carbon dioxide monitor
  - PM2.5/10 monitor
  - nitrogen dioxide monitor
4. Generate beacon report

## What's Next?
Check with advisors...
