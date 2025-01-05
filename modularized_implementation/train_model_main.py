from modularized_implementation.micro_grad import Value
from modularized_implementation.neural_network import MultiLayerPerceptron
from typing import List

def mean_squared_error(predictions: List[Value], targets: List[int]) -> Value:
    loss = Value(0.0)
    for prediction, target in zip(predictions, targets):
        # Do not use the square operator for the loss calculation. This will causes
        # in the gradient calculation. Full explanation can be found in the next cell.
        # loss += (prediction - target)**2
        loss += (prediction - target) * (prediction - target)
    return loss / len(targets)

def zero_gradients(model: MultiLayerPerceptron):
    for parameter in model.get_parameters():
        parameter.grad = 0.0
    
def update_parameters(model: MultiLayerPerceptron, learning_rate: float):
    for parameter in model.get_parameters():
        parameter.data -= learning_rate * parameter.grad

    
def train_model(model: MultiLayerPerceptron, inputs: List[List[float]], targets: List[int], num_epochs: int, learning_rate: float):
    for _ in range(num_epochs):
        predictions = []
        for input in inputs:
            predictions.append(model(input)[0])
        epoch_loss = mean_squared_error(predictions, targets)
        # The gradients should always be zeroed out before each backpropagation step.
        # Otherwise, the gradients from previous epoch will accumulate.
        zero_gradients(model=model)
        epoch_loss.backward()
        update_parameters(model=model, learning_rate=learning_rate)


if __name__ == "__main__":
    model = MultiLayerPerceptron(num_inputs=3, num_neurons_per_layer=[4, 4, 1])
    # (f1 + f2 + f3) % 2 = 0  ==>  target = 1
    # (f1 + f2 + f3) % 2 = 1  ==>  target = -1 
    inputs = [[1, 1, 1], [0, 2, 4], [-2, 4, 2], [3, 1, 3]]
    targets = [-1, 1, 1, -1]
    print(f"The targets are: {targets}")
    train_model(model=model, inputs=inputs, targets=targets, num_epochs=500, learning_rate=0.1)
    print("Final Predictions:")
    for input in inputs:
        print(model(input)[0])
    
    