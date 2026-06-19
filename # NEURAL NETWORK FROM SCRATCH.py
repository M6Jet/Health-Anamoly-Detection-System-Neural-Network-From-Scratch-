# NEURAL NETWORK FROM SCRATCH
# Health Anomaly Detetction System
# Phase 1: Core Math Functions

# Concepts: Pure Functions, scope, return values, imports
# Data: UCI Heart Disease Dataset
# Deployment: AWS Lambda

import math
import random

# Activation function: Sigmoid
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


# sigmoid(0) should give 0.5 (neutral input = neutral output)
# sigmoid(10) should give -0.99 (Strong postitive = highly activated)
# sigmoid(-10) should give -0.01 (Strong negative = barely activated)

print("--- Sigmoid Tests ---")
print(f"sigmoid(0) = {sigmoid(0)}")  # Expected: 0.5
print(f"sigmoid(10) = {round(sigmoid(10), 4)}")  # Expected: ~0.9999
print(f"sigmoid(-10) = {round(sigmoid(-10), 4)}")  # Expected: ~0.000045

def sigmoid_derivative(sigmoid_output):
    """
    Derivative of the sigmoid function.
    Used during backpropagation to calculate how much each weight
    contributed to the network's error.
    
    Parameter:
        sigmoid_output: output of the sigmoid function (float between 0 and 1)
    
    Retruns:
        float - the slope of the sigmoid at that ouput value

    Note:
        Takes sigmoid OUTPUT not raw input.
        Call sigmoid() first, then pass the result here.
    """
    return sigmoid_output * (1 - sigmoid_output)

print("\n--- Sigmoid Derivative Tests ---")
print(f"derivative at sigmoid(0) = {sigmoid_derivative(sigmoid(0))}")
print(f"derivative at sigmoid(10) = {round(sigmoid_derivative(sigmoid(10)), 6)}")
print(f"derivative at sigmoid(-10) = {round(sigmoid_derivative(sigmoid(-10)), 6)}")

def mean_squared_error(predictions, targets):
    """
    Loss function. Measures how wrong the network's predictions are.
    Lower is better. Zero means perfect predictions.
    
    Parameters:
        predictions: list of floats (network outputs, each between 0 and 1)
        targets: list of floats (correct answers, each between 0 and 1)

    Returns:
        float - average squared error across all examples

    Note:
        Both lists must be the same length.
        predictions[i] corresponds to targets[i].
    """

    # Check that both lists have the same number of items
    # This is defensive coding - catching mismatched data before it causes
    # a silent wrong answer or a confusing crash deeper in the code.
    if len(predictions) != len(targets):
        print("Error: predictions and targets muct be the same length")
        return None
    
    # List comprehension: for each index i, calculate squared error
    # (predictions[i] - targets[i]) ** 2 is the squared error for one example
    # We build a list of all squared errors than average them
    n = len(predictions)
    squared_errors = [(predictions[i] - targets[i]) ** 2 for i in range(n)]

    # sum() adds up all items in a list
    # Dividing by n gives the mean (average)
    return sum(squared_errors) / n


print("\n--- Mean Squared Error Tests ---")

# Perfect predictions - loss should be 0
perfect_preds = [1.0, 0.0, 1.0, 0.0]
perfect_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Perfect predictions loss: {mean_squared_error(perfect_preds, perfect_targets)}")  # Expected: 0.0

# Terrible predictions - loss should be high
bad_preds = [0.0, 1.0, 0.0, 1.0]
bad_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Opposite predictions loss: {mean_squared_error(bad_preds, bad_targets)}")  # Expected: 1.0

# Partial predictions - loss should be between 0 and 1
ok_preds = [0.7, 0.3, 0.6, 0.4]
ok_targets = [1.0, 0.0, 1.0, 0.0]
print(f"Partial predictions loss: {round(mean_squared_error(ok_preds, ok_targets), 4)}")  # Expected: ~0.09


def mean_squared_error_derivative(predictions, targets):
    """
    Derivative of the mean squared error loss function
    Returns a list of gradients - one per prediction.
    Each gradient tells us: push this prediction up or down, and by how much.
    
    Positive gradient means the prediction was too high - push it down.
    Negative gradient means the prediction was too low - push it up.

    Parameters:
        predictions: list of floats (network outputs)
        targets:    list of floats (correct answers)

    Returns:
        list of floats - one gradient per prediction
    """
    n = len(predictions)

    # For each prediction calculate 2 * (prediction - target) / n
    # This is the partial derivative of MSE with respect to each prediction
    gradients = [2 * (predictions[i] - targets[i]) / n for i in range(n)]

    return gradients

print("\n--- MSE Derivative Tests ---")

preds = [0.8, 0.2, 0.9, 0.1]
targets = [1.0, 0.0, 1.0, 0.0]
grads = mean_squared_error_derivative(preds, targets)

print(f"Predictions : {preds}")
print(f"Targets     : {targets}")
print(f"Gradients   : {[round(g, 4) for g in grads]}")
print("Negative gradient = prediction was too low, needs to go up")
print("Positive gradient = prediction was too high, needs to go down")

# PHASE 1 SUMMARY - ALL FUNCTIONS WORKING TOGETHER

print("\n" + "=" * 52)
print(" PHASE 1 SUMMARY - MATH FUNCTIONS")
print("=" * 52)

# Simulate a tiny network making prediciton on 4 patients
# 1 = abnmormal reading, 0 = normal reading
raw_outputs     = [2.1, -1.4, 0.8, -2.3] # Raw neuron outputs before activation
targets         = [1.0, 0.0, 1.0, 0.0]   # correct labels

# Step 1: apply sigmoid to convert raw outputs to probabilities
predictions = [sigmoid(x) for x in raw_outputs]

# Step 2: calculate how wrong we are
loss = mean_squared_error(predictions, targets)

# Step 3: calculate which direction to adjust
gradients = mean_squared_error_derivative(predictions, targets)

print(f"\nPatient Readings (raw network output -> prediction -> target):")
for i in range(len(raw_outputs)):
    pred_label = "ABNORMAL" if predictions[i] >= 0.5 else "NORMAL"
    true_label = "ABNORMAL" if targets[i] == 1.0 else "NORMAL"
    correct    = " " if pred_label == true_label else "x"
    print(f"   Patient {i+1}: {raw_outputs[i]:>5} -> {round(predictions[i], 3):.3f} -> {pred_label:<10} Actual: {true_label:<10} {correct}")

print(f"\nTotal Loss       : {round(loss, 6)}")
print(f"Gradients          : {[round(g, 4) for g in gradients]}")
print(f"\nAll 4 math functions operational.")
print("n" * 52)


# PHASE 2: THE NEURON AND THE LAYER

# Set the random seed for reproductibility
# Every run will produce the same initial weights
# This means your results will match exactly and debugging is consistent
random.seed(42)


def create_neuron(num_inputs):
    """
    Creates a single neuron with random weights and bias.

    A neuron is a dictionary with two keys:
        weights: one random float per input, between -1 and 1
        bias: one random float, between -1 and 1

    Parameters:
        num_inputs: how many inputs this neuron receives(int)

    Returns:
        dictionary with keys 'weights' and 'bias'
    
    """

    # Each weight starts as a small random number between -1 and 1
    # Starting random is important - if all weights started at the same value
    # every neuron in a layer would learn the exact same thing
    # Random initialization breaks this symmetry
    weights = [random.uniform(-1, 1) for _ in range(num_inputs)]

    # The underscore _ is a convention for a loop variable you do not need
    # We just need to run the loop num_inputs times - we do not use the index

    # Bias lets the neuron shift its activation point
    # Without bias every neuron's output would be fixed at 0.5 when all inputs are 0
    bias = random.uniform(-1, 1)

    # Return the neuron as a dictionary
    # Using a dictionary makes it easy to access weights and bias by name
    return {"weights": weights, "bias": bias}

print("--- Neuron Creation Test ---")
test_neuron = create_neuron(3)
print(f"Weights : {[round(w, 4) for w in test_neuron['weights']]}")
print(f"Bias    : {round(test_neuron['bias'], 4)}")


def neuron_forward(inputs, neuron, activation_function=sigmoid):
    """
    Runs a single forward pass through one neuron.

    Process:
        1. Multiply each input by its corresponding weight
        2. Sum all weighted inputs together
        3. Add the bias
        4. Pass through the activation function
        5. Return the output

    Parameters:
        inputs:         list of floats (one per input feature)
        neuron:         dictionary with 'weights' and 'bias' keys
        activation_function: function to apply at the end (default:sigmoid)
    
    Returns:
        tuple: (raw_sum, activated_output)
        We return both because backpropagation needs the raw sum later
    """

    # zip pairs each input with its corresponding weight
    # We multiply each pair and sum the results
    # This is called the dot product - a fundamental ML operation
    weighted_sum = sum(input_val * weight
                        for input_val, weight in zip(inputs, neuron["weights"]))
    
    # Add the bias to shift the activation threshold
    raw_sum = weighted_sum + neuron["bias"]

    # Pass through the activation function
    # Because activation_function is a parameter we could swap sigmoid
    # for any other activation function without changing this code
    output = activation_function(raw_sum)

    # Return both values as a tuple
    # raw_sum is needed during backpropagation in Phase 4
    # output is what gets passed to the next layer
    return raw_sum, output


print("\n--- Neuron Forward Pass Test ---")

# Simulate one patient reading: heart rate, blood pressure, oxygen level
# These are normalized values between 0 and 1
patient_reading = [0.8, 0.6, 0.3]

raw, output = neuron_forward(patient_reading, test_neuron)
print(f"Patient reading     : {patient_reading}")
print(f"Neuron weights      : {[round(w, 4) for w in test_neuron['weights']]}")
print(f"Neuron bias         : {round(test_neuron['bias'], 4)}")
print(f"Raw weighted sum    : {round(raw, 4)}")
print(f"Activated output    : {round(output, 4)}")
print(f"Fires strongly      : {'Yes' if output >= 0.5 else 'No'}")


def create_layer(num_neurons, num_inputs):
    """
    Creates a layer of neurons.

    A layer is a list of neuron dictionaries.
    Every neuron in the layer receives the same inputs.
    Every neuron has its own independant weights and bias.

    Parameters:
        num_neurons: how many neurons in this layer (int)
        num_inputs: how many inputs each neuron receives (int)

    Returns:
        list of neuron dictionaries
    """

    # List comprehension - create num_neurons neurons each with num_inputs weights
    # Each call to create_neuron generates fresh random weights
    # So every neuron starts different and learns different patterns
    layer = [create_neuron(num_inputs) for _ in range(num_neurons)]

    return layer

def layer_forward(inputs, layer, activation_functions=sigmoid):
    """
    Runs a forward pass through an entire layer.
    Every neuron in the layer processes the same inputs.

    Parameters:
        inputs:                 list of floats(the layer's input values)
        layer:                  list of neuron dictionaries
        activation_function:    activation function to use (default: sigmoid)

    Returns:
        tuple: (raw_sums, outputs)
        raw_sums: list of raw weighted sums before activation (for backprop)
        outputs:  list of activated outputs (passed to next layer)     
    """

    raw_sums = []
    outputs =  []

    # Run every neuron in the layer on the same inputs
    for neuron in layer:
        raw, out = neuron_forward(inputs, neuron, activation_function)
        raw_sums.append(raw)
        outputs.append(out)

    # Return both lists
    # outputs becomes the input to the next layer
    # raw_sums are stored for backpropagration
    return raw_sums, outputs

print("\n--- Layer Forward Pass Test ---")

# Create a hidden layer with 4 neurons, each receiving 3 inputs
hidden_layer = create_layer(num_neurons=4, num_inputs=3)

print(f"Layer created: 4 neurons, each with 3 weights + 1 bias")
print(f"Total parameters in this layer: {4 * (3 + 1)}")

raw_sums, layer_outputs = layer_forward(patient_reading, hidden_layer)

print(f"\nPatient reading   : {patient_reading}")
print(f"Layer outputs   :")
for i, (raw, out) in enumerate(zip(raw_sums, layer_outputs)):
    fires = "FIRES" if out >= 0.5 else "quiet"
    print(f" Neuron {i+1}: raw={round(raw,4):>7} -> output{round(out,4): .4f} ({fires})")



# PHASE 2 SUMMARY

print("\n" + "=" * 52)
print(" PHASE 2 SUMMARY - NEURON AND LAYER")
print("=" * 52)

# Build a small network structure manually to confirm everything connects
# Input layer: 3 features (heart rate, blood pressure, oxygen)
# Hidden layer: 4 neurons
# Output layer: 1 neuron (final prediction)

print("\nNetwork Architecture")
print(" Input layer : 3 features")
print(" Hidden layer: 4 neurons")
print(" Output layer: 1 neuron")
print(f" Total params : {4*(3+1) + 1*(4+1)} weights and biases")

# Create layers
hidden = create_layer(num_neurons=4, num_inputs=3)
output = create_layer(num_neurons=1, num_inputs=4)

# Run forward pass through both layers
patient = [0.75, 0.60, 0.40] #normalized heart rate, blood pressure, oxygen

_, hidden_outputs = layer_forward(patient, hidden)
_, final_outputs = layer_forward(hidden_outputs, output)

prediction = final_outputs[0]
label      = "ABNORMAL" if prediction >= 0.5 else "NORMAL"

print(f"\nPatient reading   : {patient}")
print(f"Hidden outputs      : {[round(o, 4) for o in hidden_outputs]}")
print(f"Final prediction    : {round(prediction, 4)}")
print(f"Network says        : {label}")
print(f"\nWeights are random so this prediction means nothing yet")
print(f"Training in Phase 4 is what makes the prediction trustworthy.")
print("=" * 52)



# PHASE 3: THE FULL NETWORK

def build_network(num_inputs, num_hidden, num_outputs):
    """
    CONTRACT:
    IN: num_inputs (int) - number of input features per patient
        num_hiddens (int) - number of neurons in the hidden layer
        num_outputs (int) - number of output neurons (1 for binary classification)
    OUT: network dictionary containing all layers and architecture settings
    GUARENTEE: all weights initialized randomly, seed must be set before calling

    Build a feedforward neural network with one hidden layer.
    Architecture: input -> hidden layer -> output layer -> prediction
    """

    # Build the hidden layer
    # num_hidden neurons, each receiving num_inputs values
    hidden_layer = create_layer(num_neurons=num_hidden, num_inputs=num_inputs)

    # Build the output layer
    # num_outputs neurons, each receiving num_hidden values
    # (one value per hidden neuron output)
    output_layer = create_layer(num_neurons=num_outputs, num_inputs=num_hidden)

    # Package everything into a network dictionary
    # This single object represents the entire network
    # It gets passed to every function that needs to use or modify the network
    network = {
        "hidden_layer" : hidden_layer,
        "output_layer" : output_layer,
        "num_inputs"   : num_inputs,
        "num_hidden"   : num_hidden,
        "num_outputs"  : num_outputs
    }

    # Calculate and store total parameter count
    # Parameters = (weights + bias) per neuron x number of neurons per layer
    hidden_params = num_hidden * (num_inputs + 1)
    output_params = num_outputs * (num_hidden + 1)

    network["total_params"] = hidden_params + output_params
    
    return network


print("\n--- Network Build Test ---")

# Reset seed so weights are reproducible from this point forward
random.seed(42)

# Build a network for our health detection problem
# 13 inputs = 13 features from the UCI Heart Disease Dataset
# 8 hidden neurons = enough capacity to learn meaningful patterns
# 1 output = binary prediction (normal vs abnormal)
net = build_network(num_inputs=13, num_hidden=8, num_outputs=1)

print(f"Network built successfully")
print(f"Input features  : {net['num_inputs']}")
print(f"Hidden neurons  : {net['num_hidden']}")
print(f"Output neurons  : {net['num_outputs']}")
print(f"Total parameters: {net['total_params']}")
print(f" ({net['num_hidden']} hidden neurons x ({net['num_inputs']} weights + 1 bias))")
print(f" + ({net['num_outputs']} output neuron x ({net['num_hidden']} weights + 1 bias))")


def forward_pass(network, inputs):
    """
    CONTRACT:
    IN:  network dict (from build_network)
        inputs list of floats - one per input feature, normalized 0 to 1
    OUT: cache dictionary containing:
            input       - original inputs (unchanged)
            hidden_raw  - raw weighted sums from hidden layer
            hidden_output - activated outputs from hidden layer
            output_raw  - raw weighted sums from output layer
            output_output - final activated prediciton(s)
            prediction  - single float, the network's confidence (0 to 1)
            label       - string, ABNORMAL if prediction >= 0.5 else NORMAL
    GUARANTEE: network is not modified. inputs are not modified.
    """

    # Validate input length matches network expectation
    if len(inputs) != network["num_inputs"]:
        print(f"ERROR: expected {network['num_inputs']} inputs, got {len(inputs)}")
        return None
    
    # HIDDEN LAYER FORWARD PASS
    # Pass inputs through every neuron in the hidden layer
    # layer_forward returns raw sums and activated outputs
    hidden_raw, hidden_output = layer_forward(inputs, network["hidden_layer"])

    # OUTPUT LAYER FORWARD PASS
    # The hidden layer outputs become the inputs to the output layer
    # This is the key connection - layers chain together via their outputs
    output_raw, output_output = layer_forward(hidden_output, network["output_layer"])

    # BUILD THE CACHE
    # Store every intermediate value
    # Phase 4 backpropagation needs all of these
    cache = {
        "input"     : inputs,
        "hidden_raw": hidden_raw,
        "hidden_output": hidden_output,
        "output_raw": output_raw,
        "output_output" : output_output,
        "prediction"    : output_output[0],
        "label"         : "ABNORMAL" if output_output[0] >= 0.5 else "NORMAL"
    }

    return cache


print("\n--- Forward Pass Test ---")

# Simulate one patient reading with 13 features
# In Phase 4 these will come from the real UCI dataset
# For now we use realistic normalized values
# Features: age, sex, chest pain, resting bp, cholesterol,
#           fasting blood sugar, resting ecg, max heart rate,
#           excercise angina, st depression, st slope,
#           num vessels, thal
test_patient = [0.71, 1.0, 0.67, 0.49, 0.54
                0.0, 0.5, 0.73, 0.0, 0.35,
                0.67, 0.0, 0.67]

cache = forward_pass(net, test_patient)

print(f"Input features      : {len(cache['input'])} values")
print(f"Hidden outputs      : {[round(o, 4) for o in cache ['hidden_output']]}")
print(f"Final prediction    : {round(cache['prediction'], 4)}")
print(f"Network says        : {cache['label']}")



def forward_pass_verbose(network, inputs, patient_id=1):
    """
    Identical to forward_pass but prints every intermediate calculation.
    Use this to trace exactly how data flows through the network.
    Not used during training - only for learning and debugging.

    Parameters:
        network:    network dictionary
        inputs:     list of input features
        patient_id: label for the print output (int, default 1)
    """

    print(f"\n{'-' * 52}")
    print(f" VERBOSE FORWARD PASS - Patient {patient_id}")
    print(f"{'-' * 52}")
    print(f" Raw inputs: {[round(x, 3) for x in inputs]}")

    # HIDDEN LAYER
    print(f"\n HIDDEN LAYER ({network['num_hidden']} neurons)")

    hidden_raw = []
    hidden_output = []

    for i, neuron in enumerate(network["hidden_layer"]):

        # Calculate weighted sum manually so we can print each step
        weighted_sum = sum(inp * w for inp, w in zip(inputs, neuron["weights"]))
        raw          = weighted_sum + neuron["bias"]
        out          = sigmoid(raw)
        fires        = "FIRES" if out >= 0.5 else "quiet"

        hidden_raw_append(raw)
        hidden_output.append(out)

        print(f" Neuron {i+1}: weighted_sum={round(weighted_sum, 4):>7}"
              f"+ bias={round(neuron['bias'],4):>7}"
              f"= raw={round(raw,4):>7}"
              f"-> sigmoid={round(out,4):.4f} ({fires})")
        
    # OUTPUT LAYER
    print(f"\n OUTPUT LAYER (1 neuron)")

    output_neuron = network["output_layer"][0]
    weighted_sum  = sum(h * w for * h, w in zip(hidden_output,
                                                output_neuron["weights"]))
    raw           = weighted_sum + output_neuron["bias"]
    prediction    = sigmoid(raw)
    label         = "ABNORMAL" if prediction >= 0.5 else "NORMAL"

    print(f"  Neuron 1: weighted_sum={round(weighted_sum,4):>7}"
          f"+ bias={round(output_neuron['bias'],4):>7}"
          f"= raw={round(raw,4):>7}"
          f"-> sigmoid={round(prediction,4):.4f}")
    
    # RESULT
    print(f"\n {'-' * 48}")
    print(f" PREDICTION : {round(prediction, 4)} -> {label}")
    print(f" {'-' * 48}")

    return forward_pass(network, inputs)

def batch_forward_pass(network, batch_inputs):