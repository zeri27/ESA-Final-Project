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