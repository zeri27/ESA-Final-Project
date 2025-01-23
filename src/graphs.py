import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *


def plot_values(file_name, start_date=None, end_date=None):
    data = pd.read_csv(file_name)
    parsed_file_name = parse_csv_file_name(file_name)

    # Running average
    data = pd.DataFrame(data)
    data['Value'] = data['Value'].rolling(window=12).mean().bfill()

    # Convert date strings to numbers
    data['Time'] = mdates.datestr2num(data['Time'])

    if start_date and end_date:
        data = data[(data['Time'] >= start_date) & (data['Time'] <= end_date)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(data['Time'], data['Value'], linewidth=0.7)

    # Format the x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(parsed_file_name['measurement'] + ' (' + parsed_file_name['unit'].upper() + ')')
    ax.set_title(parsed_file_name['measurement'] + ' in ' + parsed_file_name['room'] + ' on ' + parsed_file_name['floor'])

    # Show the plot
    plt.show()


if __name__ == '__main__':
    print('Plotting values...')

    start_date = mdates.date2num(datetime(2024, 11, 4))
    end_date = mdates.date2num(datetime(2024, 11, 11))

    files = get_csv_files()
    for file in files:
        print(file)
        plot_values(file, start_date, end_date)

    print('Done.')
