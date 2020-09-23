# BPEACE2 Beacon Operation Check
This file analyzes the operation of the beacons during the BPEACE2 study as part of the UTx000 extension during the second half of Spring 2020 into Summer 2020.

## Beacon 1

[b1]()

### General Notes
Time between shipment and receiving is too little - RTC must have gotten damaged. _However_, perhaps we can use the time difference between FedEx arrival time and the wrong first measurement from the beacon to calculate the offset and therefore get the correct timestamp. 

### Sensor Notes

### Debugging
- [ ] RTC: Need to check the RTC because there is no way that the beacon would have been shipped out and received/plugged in on the same day of shipping. FedEx tracking says that the beacon arrived 06/08 meaning that the battery was most likely displodged somewhere along the way. The figures shows we only had a few hours of downtime when, in reality, we should have had closer to a week. This time difference makes sense too since we should have seen the beacon logging data up until the participant moved since there is very little reason they would have unplugged it before then. The gap between the move out date and the last day of logging is just over a week which lines up pretty well.

## Beacon 5

[b5]()

### General Notes
The RTC did not come back with the beacon and checking the dates, there were problems with it from the beginning. The 06/08 data file which corresponds to the date I sent the beacon out looks to have been over-written starting at 16:22 which would be an apporximate time to when I stopped the initial calibration and packed up the device. Regardless, the device was supposed to be in transit between 06/08 and 06/09 but has data recorded continuously from 06/08 to 06/09. 

### Sensor Notes
Sensors seemed to be working just fine other than the time mix-up
- RTC: Gone - need to attach a new one

### Debugging
- [ ] RTC: Attach a new one

## Beacon 10

![b10]()

### General Notes
Shipment out and back makes sense - beacon arrived on 06/08 and last recording was on 06/05. Last recorded datapoint was on 09/10 and checking the return data file (09/23), the RTC might have been off by a few minutes, but shouldn't compromise the data.

### Sensor Notes
- CO: The data are very spotty

### Debugging
- [ ] Dates: The participant noted that they moved in the middle of the study, but there are no breaks in the data stream.
- [ ] Check T/RH data from CO sensor to see if it shows the same spottiness as the actual CO measurements.

## Beacon 13

![b13]()

### General Notes
The RTC messed up once if not twice. However, we can determine the time difference by comparing the last day's worth of data recorded by the participant compared to the first data point measured when the beacon was connected to WiFi. The data should continue to append to the participant's final datafile before the beacon gets connected to WiFi on the day that I pull data from them. So subtracting the first timestamp on the datafile after connecting to WiFi (11:01 08/28/2020) by the last timestamp on the datafile previous to this day (18:31 08/21/2020), we can get the offset (6 days, 59400 seconds). _However_, I am not certain if this holds for the entire study period or just the portion after the missing data. We need to diagnose what happened there, but for now the data should be corrected by this offset.

#### There is too much complexity in the timestamps for this beacon - I do not think we can salvage the data
Subtracting the arrival time according to FedEx by the first measurement recorded by the participant's beacon at 13:17 on 06/03, we get 6 days and 22 hours. If we add this amount back to the lst time recorded by the participant at 19:07 on 08/21, we get a time of 16:59 on 08/28 which is greater than the first timestamp for the beacon connected to WiFi at 11:01 on 08/28. My guess is that in the gap in mid-July, more RTC drift from being unplugged caused the sensor to drift too high. We **cannot** use the second time frame because we don't know when the last measured value by the participant was made nor do we have a decent estimate since they would have most likely dropped the package off in a drop box and not had the packaged picked up from their residence. Subtracting the un-connected beacon from the connected beacon timestamps will give the offset between the last measured value by the participant and the first measured value by WiFi. However this does not account for transit time.

### Sensor Notes
- RTC: Messed up once or multiple times - why that big gap?
- PM2.5: For whatever reason, things were great before the gap
- CO2: Same story as the PM2.5 sensor

### Debugging
- [ ] Dates: Check what could have caused that gap

## Beacon 15

![b15]()

### General Notes
Sending and receiving times make sense despite the odd measurement intervals. The first data point corresponds to one data point, so we might as well restrict the data to just the first interval and later. The first interval, while small, is still two nights of data. 

### Sensor Notes
- RTC: There might be a few minutes of difference between the actual and RTC time based on the first file when connected to WiFi (09/17).

### Debugging
- [ ] Dates: Might be worthwhile to investigate those time gaps

## Beacon 16

![b16]()

### General Notes
Beacon arrived on 06/09, but participant did not start recording until 07/27.

### Sensor Notes
All sensors seem to drop out for what looks like a day in the middle of the week.

### Debugging
- [ ] Beacon Sensors: check if there might be anything leading up to that event that would have caused an issue.

## Beacon 17

![b17]()

### General Notes
Shipping and receiving dates make sense and the data actually extends well into September (09/15). Not sure why the large gaps in collection, but RTC and the other sensors seem to be working fine.

### Sensor Notes
None.

### Debugging
- [ ] Dates: Might be worthwhile to investigate those time gaps

## Beacon 19

![b19]()

### General Notes
Shipment dates correspond with data collection. Sensors seem to be cooperating really well. 

### Sensor Notes
None

### Debugging
- [ ] Dates: Need to check move-out date

## Beacon 21

![b21]()

### General Notes
No shipping confirmation information, but my crude records indicate that the beacon was shipped out on the 06/08 and the last datapoint recorded was at 19:04. Sensor started recording again on 06/10 which corresponds with the typical shipping time. Last data point was recorded on 09/15 and looks like I recorded data on 09/17 - **TLDR the RTC seems fine**

### Sensor Notes
- RTC: If the RTC was off, it would have been less than a few minutes since there is no discernable problems from reading the return (09/17) data file.

### Debugging
None

## Beacon 22

![b22]()

### General Notes
Seems the participant had their beacon plugged in up until they moved and then plugged the device in again at their new location since I did not pick the device up until 09/03. I downloaded the data from the device on the same day which is why there are no discernable data points after 09/03 corresponding to when I would have plugged in the device at my apartment. The last recorded data point by the participant was at 9:35 on 09/03 and then it seems I plugged the device in at 18:53 that day to pull the data off. 

### Sensor Notes
- PM: Seems to have been troubles with the sensor during the main study period, but the sensor seemed to be working well during the second period after the participant moved.

### Debugging
None

## Beacon 24

![b24]()

### General Notes
Sensor was shipped out just after 19:00 on 06/08 (last recorded measurement at 19:04) and arrived at the participant's location on 06/10 at 14:04. Participant did not start logging data until the next day though if all the times are correct. The measurements on 06/08 after 19:00 concern me since I believe the dropbox pick up is done at 18:00 although it could be 20:00, but that seems unlikely. Also the only data collected on 06/08 is from 18:19 to 19:04 even though on 06/07 the sensor recorded up to 23:59. 

### Sensor Notes
- Beacon sensors: There is a weird blip on 08/17 when all the sensors go down. Checking the data, the beacon was offline from 19:30 on 08/16 until 21:10 on 08/17. Perhaps the device was unplugged? Not sure if there is a way to find out what happened. 

### Debugging
- [ ] Dates (1): Try and determine what might have happened during the ~1 day of downtime
- [ ] Dates (2): Can shipment could occur after 19:00?

## Beacon 25

![b25]()

### General Notes
Beacon was shipped on 06/03 and arrived the next day at 06/04 and started recording at 11:54 (arrived at 11:31 - I knew I liked this participant).

### Sensor Notes
None

### Debugging
None

## Beacon 26

![b26]()

### General Notes
Just like #24, the beacon last recorded data with me at 19:05 on 06/08 which was the day it was shipped out. It also recorded data up until 23:59 on 06/07 and then nothing on 06/08 until 18:35. Shipment arrived on 06/10 but participant did not start recording until 06/20 at 18:43 and kept recording until 8:46 on 09/08 which is the day they shipped it back. 

### Sensor Notes
GPS: We lose a good chunk of GPS data near the end of the beacon recording period most likely because the participant had their appointment with Melissa around then and deactivated their Beiwe app.

### Debugging
- [ ] Dates: Can shipment could occur after 19:00?

## Beacon 28

![b28]()

### General Notes
Shipment dates correspond decently well although participant didn't start logging data until much later (after they moved at least).

### Sensor Notes
- Light: No light sensor data - must have gotten disconnected
- NO2: Data are sparse 

### Debugging
- [ ] Light: Check connections/wires. Might need to crimp new ones. 
- [ ] NO2: Check T/RH data

## Beacon 29

![b29]()

### General Notes
No tracking information but my records indicate that it was shipped out on 06/10. The beacon stopped recording at 12:13 on 06/10 and started again at 12:01 on 06/11 which makes sense given most of the beacons were sent with 2-day shipping but often arrived early. 

### Sensor Notes
- NO2: I didn't think this beacon had an NO2 sensor, but I suppose it does! I think up to and including 30 do. Regardless, the sensor performs poorly.

### Debugging
- [ ] NO2: Check to see how the T/RH data looks from this sensor.

## Beacon 32

![b32]()

### General Notes
Participant plugged in the device late and then unplugged before they moved and did not plug the device back in.

### Sensor Notes
NO2 seems a bit spotty, but not nearly as bad as some of the other beacons.

### Debugging
None

## Beacon 34

[b34]()

### General Notes
The gap between sending and receiving the beacon is about 1 day which is _surprisingly_ correct for this beacon. The beacon last recorded values at 13:18 on 06/10 and was shipped out at 18:00 that day. The package was delivered at 12:01 on 06/11 according to FedEx and began recording at 13:41 on 06/12.

### Sensor Notes
- CO: This beacon has no NO2 sensor; the NO2 readings are really CO readings since the USB channels reset when only one is plugged in.

### Debugging
- [ ] Dates: Need to check if the move-out date is correct or even occurred since there looks to be no break in the measurements. 

## Beacon 36

![b36]()

### General Notes
Shipment details seem to correspond with data collection.

### Sensor Notes
- NO2: Some spottiness here and there but generally very good.

### Debugging
- [ ] Dates: Need to check if the move-out date is correct or even occurred since there looks to be no break in the measurements. 

## Beacon 38

![b38]()

### General Notes
Records indicate the beacon shipped on 06/10, but did not start recording data until 06/15 which could be explained by the location of the participant since they lived outside Austin. No shipping info to check against. 

### Sensor Notes
- Beacon sensors: Pretty spotty which might just be because of how the beacons were plugged in that perhaps they had too low of power. The gaps also don't align with the move out date.
- NO2: Again, didn't think these beacons had an NO2 sensor but it seems to be recording poorly compared to the rest.

### Debugging
- [ ] Dates: Need to check if the move-out date is correct or even occurred since there looks to be no break in the measurements. 
- [ ] NO2: Check measurement values for T/RH.

## Beacon 40

![b22]()

### General Notes
Two conflicting shipment dates, but the participant did not start recording until 06/30 and both shipment dates were before then so nothing majorly conflicting. Otherwise spotty connection with the sensors that does not match up with the move-out date. 

### Sensor Notes
- Beacon Py3 sensors: No data from sensors running python3 even during the data retrieval process.
- GPS: No data

### Debugging
- [ ] Dates: Need to check if the move-out date is correct or even occurred since there looks to be no break in the measurements. 
- [ ] GPS: Check again for GPS data
- [ ] Py3 sensors: Check RPi connections to see why the sensors did/are not collecting. 

## Beacon 46

![b46]()

### General Notes
Shipment dates match with measurment dates. However, we only get a few hours on 06/17 so we should start including data starting 06/20.

### Sensor Notes
- PM: Seems to be exhibiting problems and no GPS data...

### Debugging
- [ ] Dates: Is there a way to find out why the device came online for a short while?
- [ ] GPS: Check again if GPS data are available

