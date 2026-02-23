from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from static import FEATURES
from utils import read_dataframe

SCALED_FEATURES = [f'{c}_n' for c in FEATURES]

def read_standarized_data() -> pd.DataFrame:
    iris_df = read_dataframe()
    # TODO: Normalize the features

    return iris_df

def run_elbow_method():
    standardized_df = read_standarized_data()
    mse_data = []
    # TODO: Run KMeans for k = 1 to 10 and calculate the MSE (inertia) for each k

    # TODO: Visualize the result with a line plot (k on x-axis and MSE on y-axis)


def run_kmeans(k: int = 3):
    standardized_df = read_standarized_data()
    # TODO: Run KMeans with k clusters and get the cluster labels for each data point

    # TODO: Visualize the result with a scatter plot (Petal_length on x-axis and Sepal_length on y-axis, color by cluster)


if __name__ == '__main__':
    print('[Q1][Part 1] The normalized dataframe looks like this:')
    print(read_standarized_data().head())
    print('[Q1][Part 2] Plot a line chart to show how to find the best K using the Elbow method')
    run_elbow_method()
    print('[Q1][Part 3] Visualize K-means with k=3')
    run_kmeans(k=3)
