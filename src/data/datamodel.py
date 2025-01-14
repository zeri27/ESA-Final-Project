import pandas as pd

# Read CSV files
co2_data = pd.read_csv("../../data/floor1_meetingroom_co2meter_co2_ppm.csv")
humidity_data = pd.read_csv("../../data/floor1_meetingroom_co2meter_humidity_perc.csv")
temperature_data = pd.read_csv("../../data/floor1_meetingroom_co2meter_temp_c.csv")

# Convert 'Time' column to proper format
co2_data['Time'] = pd.to_datetime(co2_data['Time'])
humidity_data['Time'] = pd.to_datetime(humidity_data['Time'])
temperature_data['Time'] = pd.to_datetime(temperature_data['Time'])

# Merge DataFrames on 'Time'
merged_data = pd.merge(co2_data, humidity_data, on="Time", how="outer", suffixes=('_CO2', '_Humidity'))
merged_data = pd.merge(merged_data, temperature_data, on="Time", how="outer")
merged_data.rename(columns={"Value_CO2": "CO2_PPM", "Value_Humidity": "Humidity", "Value": "Temperature"}, inplace=True)
merged_data['Date'] = merged_data['Time'].dt.date
merged_data['Time'] = merged_data['Time'].dt.time
merged_data = merged_data[['Date', 'Time', 'CO2_PPM', 'Humidity', 'Temperature']]

# print(merged_data.head())
# print(merged_data.shape[0])

merged_data = merged_data.dropna()
print("Data Ready")
# print(merged_data.shape[0])

data = merged_data
