import torch
from static import FEATURES, TARGET, TARGET_CLASS_DICT
from utils import load_tensors


def run_prediction(_model, x_tensor):
    # Given a model and input, predict the corresponding output
    with torch.no_grad():
        test_predict = _model(x_tensor)
        _, predicted_classes = torch.max(test_predict, 1)  # Find the class with the highest score
    return predicted_classes

def calc_accuracy(_model, x_tensor, y_tensor):
    predicted_classes = run_prediction(_model, x_tensor)
    # Accuracy = Number of correct prediction / Number of items to be predicted
    return (predicted_classes == y_tensor).sum().item() / y_tensor.size(0)


def train_model(x_tensor, y_tensor) -> torch.nn.Sequential:
    torch.manual_seed(5963)
    # TODO: Configure your model structure
    # Usage of torch.nn.linear from Gemini 3 - fast via https://gemini.google.com/
    # Prompt: You are a machine learning expert. Now I am going to use Iris dataset to build a neural network. Please give me some hints on how I can use torch.nn.linear() to configure my model structure (e.g., the number of neurons in Linear(), or if I need to use RELU, dropout). Your language style should be understandable for non-tech people. Don't write the actual code for me because I want to challenge myself first. :)
    # Gemini told me:
    # For Iris, try 8, 16, or 32 neurons in my hidden layers. Otherwise, overfitting would occur.
    # We really only need one or two hidden layers for this dataset. It's a small task, so don't over-engineer it!
    # The dataset is so small and simple that Dropout might actually make it harder for the model to learn anything at all.
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, len(TARGET_CLASS_DICT))  # len(TARGET_CLASS_DICT) is 3 (Setosa, Versicolor, Verginica)
    )
    # Cross Entropy Loss is used for classification
    loss_function = torch.nn.CrossEntropyLoss()

    # TODO: Configure your hyper-parameters
    num_epochs = 500
    learning_rate = 0.01
    # TODO: Configurable: optimizer
    # Reference: https://runebook.dev/zh/docs/pytorch/generated/torch.optim.sgd/torch.optim.SGD
    optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr=learning_rate, weight_decay = 0.0001, nesterov=True)

    # Start training (do not change this unless you know what you are doing)
    for epoch in range(num_epochs):
        optimizer.zero_grad()                           # Resets gradients
        train_predict = model(x_tensor)                 # Make a prediction
        loss = loss_function(train_predict, y_tensor)   # Calculate loss
        loss.backward()                                 # Calculate gradient
        optimizer.step()                                # Update weights using the gradient
        if (epoch + 1) % (num_epochs/10) == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


if __name__ == '__main__':
    train_x_tensor, test_x_tensor, train_y_tensor, test_y_tensor = load_tensors()
    your_model = train_model(train_x_tensor, train_y_tensor)
    print(f'[Q3] Train accuracy: {calc_accuracy(your_model, train_x_tensor, train_y_tensor):.2%}')
    print(f'[Q3] Test accuracy: {calc_accuracy(your_model, test_x_tensor, test_y_tensor):.2%}')
    model_file_name = 'q3.pth'
    torch.save(your_model, model_file_name)
    print(f'Exported model as {model_file_name}')
    guess = [2, 5, 3, 6]
    predict_species = TARGET_CLASS_DICT[run_prediction(your_model, torch.Tensor(guess).view(1, -1))[0].item()]
    msg = 'CORRECT' if predict_species == 'Iris-virginica' else 'WRONG'
    print(f'[{msg}] Prediction of {dict(zip(FEATURES, guess))} from your model: {predict_species}')