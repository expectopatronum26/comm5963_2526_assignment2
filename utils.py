import os
import io
import zipfile
import sklearn
import pandas as pd
import requests
import torch
from static import FEATURES, TARGET, TARGET_CLASS_DICT

def read_dataframe() -> pd.DataFrame:
    # Data Reference: https://archive.ics.uci.edu/dataset/53/iris
    iris_zip_url = 'https://archive.ics.uci.edu/static/public/53/iris.zip'
    csv_file_name = 'iris.csv'
    # Download the Excel spreadsheet if it does not exist
    if not os.path.exists(csv_file_name):
        print(f'Downloading: {iris_zip_url}')
        response = requests.get(iris_zip_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open('iris.data') as csv_file:
                 with open(csv_file_name, 'wb') as f:
                    f.write(csv_file.read())
    else:
        print(f'Reusing previously downloaded file: {csv_file_name}')
    # Read the dataframe
    df = pd.read_csv(csv_file_name, header=None,
                     names=['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Species_name'])
    return df


def load_train_test_datasets():
    df = read_dataframe()
    # Dataset split: 80% Train vs 20% Test
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=5963)
    return train_x, train_y, test_x, test_y


def load_tensors():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # The Y is in string, we need to convert it to a number before passing into our Neural Network
    class_to_id_dict = {v: k for k, v in TARGET_CLASS_DICT.items()}
    train_y = train_y.map(class_to_id_dict)
    test_y = test_y.map(class_to_id_dict)
    # Convert to PyTorch tensors
    _train_x_tensor = torch.tensor(train_x.to_numpy().copy(), dtype=torch.float32)
    _test_x_tensor = torch.tensor(test_x.to_numpy().copy(), dtype=torch.float32)
    _train_y_tensor = torch.tensor(train_y.to_numpy().copy(), dtype=torch.long)
    _test_y_tensor = torch.tensor(test_y.to_numpy().copy(), dtype=torch.long)
    return _train_x_tensor, _test_x_tensor, _train_y_tensor, _test_y_tensor

if __name__ == '__main__':
    df_iris = read_dataframe()
    print(df_iris)