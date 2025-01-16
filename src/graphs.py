import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os


def get_csv_files():
    file_paths = []
    for file_name in os.listdir('../data'):
        if file_name.endswith('.csv'):
            file_path = os.path.join('../data', file_name)
            if os.path.isfile(file_path):
                file_paths.append(file_path)
    return file_paths


def parse_csv_file_name(file_name):
    without_ext = os.path.splitext(os.path.basename(file_name))[0]

    components = without_ext.split('_')
    return {
        name: component for name, component in zip(['floor', 'room', 'meter', 'measurement', 'unit'], components)
    }


def plot_values(file_name):
    data = pd.read_csv(file_name)
    parsed_file_name = parse_csv_file_name(file_name)

    # Convert date strings to numbers
    date_nums = mdates.datestr2num(data['Time'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(date_nums, data['Value'], linewidth=0.7)

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

    files = get_csv_files()
    for file in files:
        print(file)
        plot_values(file)

    print('Done.')
