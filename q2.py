import sklearn
import matplotlib.pyplot as plt

from utils import load_train_test_datasets
from static import FEATURES, TARGET


def run_decision_tree():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    model = None
    # TODO: Run a classification by constructing a decision tree (Please set the random_state to 5963)

    # TODO: Print the train and test accuracy of the model

    return model

def show_decision_tree(model_from_part1):
    # TODO: Visualize the decision tree
    model_from_part1


def run_random_forest():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # TODO: Run a classification by constructing a random forest (Please set the random_state to 5963)

    # TODO: Print the train and test accuracy of the model



if __name__ == '__main__':
    print('[Q2][Part 1] Run Decision Tree')
    model = run_decision_tree()
    print('[Q2][Part 2] Visualize the Decision Tree')
    show_decision_tree(model_from_part1=model)
    print('[Q2][Part 3] Run Random Forest')
    run_random_forest()
