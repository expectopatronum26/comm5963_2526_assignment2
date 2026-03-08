import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from utils import load_train_test_datasets
from static import FEATURES, TARGET


def run_decision_tree():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    model = None
    # TODO: Run a classification by constructing a decision tree (Please set the random_state to 5963)
    # Create and train the Decision Tree Classifier
    model = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, random_state = 5963)
    model.fit(X = train_x, y = train_y)

    # Train & test accuracy
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)

    # TODO: Print the train and test accuracy of the model
    print(f'Train accuracy: {train_score:.2%}')
    print(f'Test accuracy: {test_score:.2%}')

    return model

def show_decision_tree(model_from_part1):
    # TODO: Visualize the decision tree
    features = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
    plt.figure(figsize=(16, 6))
    plot_tree(
        model,
        feature_names=features,
        class_names=model.classes_,
        filled=True,
        fontsize=18
    )
    plt.title(f'Survival prediction by {"+".join(features)}')
    plt.show()

def run_random_forest():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # TODO: Run a classification by constructing a random forest (Please set the random_state to 5963)
    from sklearn.ensemble import RandomForestClassifier

    # Train a Random Forest
    model = RandomForestClassifier(
        n_estimators = 4,  # How many random trees we use
        criterion = 'gini',  # Aim: Lower Gini impurtiy
        max_depth = 2,  # How many cuts
        random_state = 5963  # Control randomness (optional)
    )
    model.fit(X = train_x, y = train_y)


    # TODO: Print the train and test accuracy of the model
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    print(f'Train Accuracy: {train_score:.2%}')
    print(f'Test accuracy: {test_score:.2%}')



if __name__ == '__main__':
    print('[Q2][Part 1] Run Decision Tree')
    model = run_decision_tree()
    print('[Q2][Part 2] Visualize the Decision Tree')
    show_decision_tree(model_from_part1=model)
    print('[Q2][Part 3] Run Random Forest')
    run_random_forest()
