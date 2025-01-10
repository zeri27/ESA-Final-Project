import pandas as pd
import matplotlib.pyplot as plt

# files = [
#   'floor1_meetingroom_co2meter_co2_ppm.csv',
#   'floor1_meetingroom_co2meter_humidity_perc.csv',
#   'floor1_meetingroom_co2meter_temp_c.csv',

#   'floor2_meetingroom_co2meter_co2_ppm.csv',
#   'floor2_meetingroom_co2meter_humidity_perc.csv',
#   'floor2_meetingroom_co2meter_temp_c.csv',

#   'floor1_office_heater_consumption_j.csv',
#   'floor1_office_heater_power_w.csv',
#   'floor1_office_co2meter_co2_ppm.csv',
#   'floor1_office_co2meter_humidity_perc.csv',
#   'floor1_office_co2meter_temp_c.csv'
# ]

def readFloor(prefix: str):
  def getPath(name: str) -> str:
    return f"../data/{prefix}_co2meter_{name}"

  co2 = pd.read_csv(getPath('co2_ppm.csv'))
  humidity = pd.read_csv(getPath('humidity_perc.csv'))
  temp = pd.read_csv(getPath('temp_c.csv'))
  merged = pd.merge(co2, humidity, on = 'Time') #outter join?
  merged = pd.merge(merged, temp, on = 'Time')
  columns = ['co2', 'humidity', 'temp']
  merged.columns = ['Time'] + columns
  merged['combined'] = merged[columns].apply(list, axis = 1)
  merged = merged.drop(columns = columns)
  print(merged.head())
  return merged

f1meetingRoom = readFloor("floor1_meetingroom")
f1office = readFloor("floor1_office")
f2meetingRoom = readFloor("floor2_meetingroom")

merged = pd.merge(f1meetingRoom, f1office, on = 'Time')
merged = pd.merge(merged, f2meetingRoom, on = 'Time')
columns = ['f1meetingRoom', 'f1office', 'f2meetingRoom']
merged.columns = ['Time'] + columns
merged['combined'] = merged[columns].apply(list, axis = 1)
merged = merged.drop(columns = columns)
print(merged['combined'][0])
flattened = merged.explode('combined', ignore_index = True)
print(flattened.head())

a = flattened.to_numpy()
print(a[0])