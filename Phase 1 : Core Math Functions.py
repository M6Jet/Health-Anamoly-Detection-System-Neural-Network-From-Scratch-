# NEURAL NETWORK FROM SCRATCH
# Health Anomaly Detection System
# Phase 1: Core Math Functions
# Concepts: pure functions, scope, return values, imports
# Data: UCI Heart Disease Dataset
# Deployment: AWS Lambda

import math


def sigmoid(x):
    """
    Activation function. Maps any real number to a value between 0 and 1.
    Used at the output of every neuron to represent activation strength.
    
    Parameter:
        x: any real number (float or int)
    
    Returns:
        float between 0 and 1
    """
    return 1 / (1 + math.exp(-x))



print("--- Sigmoid Tests ---")
print(f"sigmoid(0)   = {sigmoid(0)}")
print(f"sigmoid(10)  = {round(sigmoid(10), 4)}")
print(f"sigmoid(-10) = {round(sigmoid(-10), 4)}")

def sigmoid_derivative(sigmoid_output):
    """
    Derivative of the sigmoid function.
    Used during backpropagation to calculate how much each weight
    contributed to the network's error.
    
    Parameter:
        sigmoid_output: the output of a sigmoid function (float between 0 and 1)
        
    Returns:
        float — the slope of the sigmoid at that output value
        
    Note:
        Takes sigmoid OUTPUT not raw input.
        Call sigmoid() first, then pass the result here.
    """
    return sigmoid_output * (1 - sigmoid_output)


print("\n--- Sigmoid Derivative Tests ---")
print(f"derivative at sigmoid(0)   = {sigmoid_derivative(sigmoid(0))}")
print(f"derivative at sigmoid(10)  = {round(sigmoid_derivative(sigmoid(10)), 6)}")
print(f"derivative at sigmoid(-10) = {round(sigmoid_derivative(sigmoid(-10)), 6)}")


def mean_squared_error(predictions, targets):
    """
    Loss function. Measures how wrong the network's predictions are.
    Lower is better. Zero means perfect predictions.
    
    Parameters:
        predictions: list of floats (network outputs, each between 0 and 1)
        targets:     list of floats (correct answers, each 0 or 1)
        
    Returns:
        float — average squared error across all examples
        
    Note:
        Both lists must be the same length.
        predictions[i] corresponds to targets[i].
    """


    if len(predictions) != len(targets):
        print("ERROR: predictions and targets must be the same length")
        return None


    n = len(predictions)
    squared_errors = [(predictions[i] - targets[i]) ** 2 for i in range(n)]

    return sum(squared_errors) / n


print("\n--- Mean Squared Error Tests ---")

perfect_preds   = [1.0, 0.0, 1.0, 0.0]
perfect_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Perfect predictions loss : {mean_squared_error(perfect_preds, perfect_targets)}")

bad_preds   = [0.0, 1.0, 0.0, 1.0]
bad_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Opposite predictions loss: {mean_squared_error(bad_preds, bad_targets)}")

ok_preds   = [0.7, 0.3, 0.6, 0.4]
ok_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Partial predictions loss : {round(mean_squared_error(ok_preds, ok_targets), 4)}")


def mean_squared_error_derivative(predictions, targets):
    """
    Derivative of the mean squared error loss function.
    Returns a list of gradients — one per prediction.
    Each gradient tells us: push this prediction up or down, and by how much.
    
    Positive gradient means the prediction was too high — push it down.
    Negative gradient means the prediction was too low — push it up.
    
    Parameters:
        predictions: list of floats (network outputs)
        targets:     list of floats (correct answers)
        
    Returns:
        list of floats — one gradient per prediction
    """
    n = len(predictions)

    gradients = [2 * (predictions[i] - targets[i]) / n for i in range(n)]

    return gradients


print("\n--- MSE Derivative Tests ---")

preds   = [0.8, 0.2, 0.9, 0.1]
targets = [1.0, 0.0, 1.0, 0.0]
grads   = mean_squared_error_derivative(preds, targets)

print(f"Predictions : {preds}")
print(f"Targets     : {targets}")
print(f"Gradients   : {[round(g, 4) for g in grads]}")
print("Negative gradient = prediction was too low, needs to go up")
print("Positive gradient = prediction was too high, needs to go down")


print("\n" + "=" * 52)
print(" PHASE 1 SUMMARY — MATH FUNCTIONS")
print("=" * 52)

raw_outputs     = [2.1, -1.4, 0.8, -2.3]    # raw neuron outputs before activation
targets         = [1.0,  0.0, 1.0,  0.0]    # correct labels

predictions = [sigmoid(x) for x in raw_outputs]

loss = mean_squared_error(predictions, targets)

gradients = mean_squared_error_derivative(predictions, targets)

print(f"\nPatient Readings (raw network output → prediction → target):")
for i in range(len(raw_outputs)):
    pred_label = "ABNORMAL" if predictions[i] >= 0.5 else "NORMAL"
    true_label = "ABNORMAL" if targets[i] == 1.0 else "NORMAL"
    correct    = "✓" if pred_label == true_label else "✗"
    print(f"  Patient {i+1}: {raw_outputs[i]:>5} → {round(predictions[i], 3):.3f} → {pred_label:<10} | Actual: {true_label:<10} {correct}")

print(f"\nTotal Loss     : {round(loss, 6)}")
print(f"Gradients      : {[round(g, 4) for g in gradients]}")
print(f"\nAll 4 math functions operational.")
print("=" * 52)

