from utils import *
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
rolling_avg = 25
n_clusters = 2
algorithm = "lloyd"
measurement = ['co2', 'temp']
s_d = mdates.date2num(datetime(2024, 9, 21))    # One week: 2024-11-04
e_d = mdates.date2num(datetime(2024, 12, 21))   # One week: 2024-11-11


def preprocess_data(file_name, data, start_date=None, end_date=None, rolling_avg_window=None):
    """
    Preprocess data from a file into a workable data frame.
    :param file_name: Name of file to fetch data from
    :param data: Data in the file
    :param start_date: (Optional) Start date of data
    :param end_date: (Optional) End date of data
    :param rolling_avg_window: (Optional) Window size of rolling average
    :return: Preprocessed data frame.
    """
    parsed_name = parse_csv_file_name(file_name)

    # Extract data from start date to end date, if applicable
    if start_date and end_date:
        conv_times = mdates.datestr2num(data['Time'])
        data = data[(conv_times >= start_date) & (conv_times <= end_date)]
        if len(data) == 0:
            return None

    # Convert to data frame
    df = pd.DataFrame(data)

    # Calculate rolling averages, if applicable
    if rolling_avg_window:
        df['Value'] = df['Value'].rolling(window=rolling_avg_window).mean().bfill()

    df.rename(columns={'Value': f'value_{parsed_name["measurement"]}'}, inplace=True)

    # Bin data in five-minute means
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.resample('5min').mean()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.reset_index(inplace=True)

    # Add columns measurement, floor, room
    df['measurement'] = parsed_name['measurement']
    df['floor'] = parsed_name['floor']
    df['room'] = parsed_name['room']

    # Add weekday and time_of_day
    df['weekday'] = df['Time'].dt.weekday
    df['time_of_day'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute  # (time in minutes from midnight)

    return df


def set_up_pipeline(active_measurements):
    """
    Sets up the pipeline for clustering.
    :param active_measurements: List of units of measurement to fetch data of
    :return: Preprocessor and pipeline objects.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), [f'value_{m}' for m in active_measurements] + ['time_of_day']),
            ('cat', OneHotEncoder(), ['weekday'])
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KMeans(n_clusters=n_clusters, random_state=42, algorithm=algorithm))
        # ('classifier', MiniBatchKMeans(n_clusters=n_clusters, random_state=42))   # For Mini-Batch K-Means
    ])

    return preprocessor, pipeline


def run_pipeline(preprocessor, pipeline, df):
    """
    Runs the given pipeline with a given data frame and plots the cluster results.
    :param preprocessor: Preprocessor object
    :param pipeline: Pipeline object
    :param df: Data frame
    """
    pipeline.fit(df)

    df['Cluster'] = pipeline.predict(df)

    # If wished, perform PCA
    ### PCA START ###

    # X_transformed = pipeline.named_steps['preprocessor'].transform(df)
    #
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_transformed)
    #
    # # Create a DataFrame for PCA components
    # pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    #
    # # Get the loadings of the original features
    # loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    #
    # # Create a DataFrame for loadings
    # loading_df = pd.DataFrame(loadings, index=preprocessor.get_feature_names_out(), columns=['PC1', 'PC2'])
    #
    # # Display the loadings
    # print(loading_df)
    #
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
    # plt.title('K-Means Clustering of Room Occupancy')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.colorbar(label='Cluster')
    # plt.show()

    ### PCA END

    # Plot the fitted data frame
    plot_df(df)


def plot_df(df):
    """
    Plot the given data frame.
    :param df: Processed and clustered data frame to plot
    """
    def format_time(minutes):
        """
        Formats a number in minutes to a string %H:%M.
        :param minutes: Number of minutes
        :return: String with time formatted as %H:%M.
        """
        hours = minutes // 60
        mins = minutes % 60
        return f'{hours:02}:{mins:02}'

    # Apply names of the weekdays to data, and order according to weekdays (from Monday to Sunday)
    df['weekday'] = df['weekday'].apply(lambda x: weekdays[x])
    df.set_index('weekday', inplace=True)
    df = df.loc[weekdays]
    df.reset_index(inplace=True)
    plt.scatter(df['weekday'], df['time_of_day'], c=df['Cluster'], cmap='viridis')
    plt.gcf().autofmt_xdate()

    # Set weekday as x-axis
    plt.xticks(ticks=range(len(weekdays)), labels=weekdays)

    # Set time of day as y-axis
    y_ticks = range(0, 24 * 60, 60)
    plt.yticks(y_ticks, [format_time(tick) for tick in y_ticks])

    plt.xlabel('Weekday')
    plt.ylabel('Time of day')
    plt.title(f'K-Means Clustering {measurement}' + (f' [{mdates.num2date(s_d).strftime("%Y-%m-%d")} ~ {mdates.num2date(e_d).strftime("%Y-%m-%d")}]' if s_d is not None and e_d is not None else '') + f'\n(n_clusters={n_clusters}; random_seed=42; rolling_avg_window={rolling_avg})')
    # plt.title(f'Mini-Batch K-Means Clustering\n(n_clusters={n_clusters}; random_seed=42; rolling_avg_window={rolling_avg})')  # For Mini-Batch K-Means
    plt.show()


if __name__ == "__main__":
    files = [f for f in get_csv_files() if parse_csv_file_name(f)['measurement'] in measurement]

    dfs = {}

    # Enumerate through data files and preprocess them
    for i, f in enumerate(files, start=1):
        print(f'Processing [{i}/{len(files)}] {f}')
        d = pd.read_csv(f)
        d = preprocess_data(f, d, s_d, e_d, rolling_avg_window=rolling_avg)
        current_measurement = parse_csv_file_name(f)['measurement']
        if d is not None:
            if current_measurement not in dfs:
                dfs[current_measurement] = [d]
            else:
                dfs[current_measurement] += [d]
        else:
            print('Dropped')

    # Concatenate to a single data frame
    print('Concatenating data')
    df = None

    for k, v in dfs.items():
        combined = pd.concat(v, ignore_index=True)

        combined = combined.groupby(['Time'], as_index=False).agg({
            'time_of_day': 'first',
            'weekday': 'first',
            f'value_{k}': 'mean'
        }).reset_index()

        if df is None:
            df = combined
        else:
            df[f'value_{k}'] = combined[f'value_{k}']

    # Drop N/A values
    df.dropna(inplace=True)

    print('df:')
    print(df)

    # Initialize and run pipeline
    print('Setting up pipeline')
    preprocessor, pipeline = set_up_pipeline(dfs.keys())
    print('Running pipeline')
    run_pipeline(preprocessor, pipeline, df)
    print('Done.')
