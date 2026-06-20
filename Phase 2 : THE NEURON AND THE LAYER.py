import math
import random

# PHASE 2: THE NEURON AND THE LAYER

random.seed(42)


def create_neuron(num_inputs):
    """
    Creates a single neuron with random weights and bias.
    
    A neuron is a dictionary with two keys:
        weights: one random float per input, between -1 and 1
        bias: one random float, between -1 and 1
    
    Parameters:
        num_inputs: how many inputs this neuron receives (int)
        
    Returns:
        dictionary with keys 'weights' and 'bias'
    """

    weights = [random.uniform(-1, 1) for _ in range(num_inputs)]

    bias = random.uniform(-1, 1)

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
        inputs:              list of floats (one per input feature)
        neuron:              dictionary with 'weights' and 'bias' keys
        activation_function: function to apply at the end (default: sigmoid)
        
    Returns:
        tuple: (raw_sum, activated_output)
        We return both because backpropagation needs the raw sum later
    """


    weighted_sum = sum(input_val * weight
                       for input_val, weight in zip(inputs, neuron["weights"]))


    raw_sum = weighted_sum + neuron["bias"]

    output = activation_function(raw_sum)

    return raw_sum, output


print("\n--- Neuron Forward Pass Test ---")

patient_reading = [0.8, 0.6, 0.3]

raw, output = neuron_forward(patient_reading, test_neuron)
print(f"Patient reading  : {patient_reading}")
print(f"Neuron weights   : {[round(w, 4) for w in test_neuron['weights']]}")
print(f"Neuron bias      : {round(test_neuron['bias'], 4)}")
print(f"Raw weighted sum : {round(raw, 4)}")
print(f"Activated output : {round(output, 4)}")
print(f"Fires strongly   : {'Yes' if output >= 0.5 else 'No'}")


def create_layer(num_neurons, num_inputs):
    """
    Creates a layer of neurons.
    
    A layer is a list of neuron dictionaries.
    Every neuron in the layer receives the same inputs.
    Every neuron has its own independent weights and bias.
    
    Parameters:
        num_neurons: how many neurons in this layer (int)
        num_inputs:  how many inputs each neuron receives (int)
        
    Returns:
        list of neuron dictionaries
    """


    layer = [create_neuron(num_inputs) for _ in range(num_neurons)]

    return layer


def layer_forward(inputs, layer, activation_function=sigmoid):
    """
    Runs a forward pass through an entire layer.
    Every neuron in the layer processes the same inputs.
    
    Parameters:
        inputs:              list of floats (the layer's input values)
        layer:               list of neuron dictionaries
        activation_function: activation function to use (default: sigmoid)
        
    Returns:
        tuple: (raw_sums, outputs)
            raw_sums: list of raw weighted sums before activation (for backprop)
            outputs:  list of activated outputs (passed to next layer)
    """

    raw_sums = []
    outputs  = []

    for neuron in layer:
        raw, out = neuron_forward(inputs, neuron, activation_function)
        raw_sums.append(raw)
        outputs.append(out)

    return raw_sums, outputs


print("\n--- Layer Forward Pass Test ---")

hidden_layer = create_layer(num_neurons=4, num_inputs=3)

print(f"Layer created: 4 neurons, each with 3 weights + 1 bias")
print(f"Total parameters in this layer: {4 * (3 + 1)}")

raw_sums, layer_outputs = layer_forward(patient_reading, hidden_layer)

print(f"\nPatient reading  : {patient_reading}")
print(f"Layer outputs    :")
for i, (raw, out) in enumerate(zip(raw_sums, layer_outputs)):
    fires = "FIRES" if out >= 0.5 else "quiet"
    print(f"  Neuron {i+1}: raw={round(raw,4):>7} → output={round(out,4):.4f} ({fires})")


print("\n" + "=" * 52)
print(" PHASE 2 SUMMARY — NEURON AND LAYER")
print("=" * 52)


print("\nNetwork Architecture:")
print("  Input  layer : 3 features")
print("  Hidden layer : 4 neurons")
print("  Output layer : 1 neuron")
print(f"  Total params : {4*(3+1) + 1*(4+1)} weights and biases")

hidden = create_layer(num_neurons=4, num_inputs=3)
output = create_layer(num_neurons=1, num_inputs=4)

patient = [0.75, 0.60, 0.40]   # normalized heart rate, blood pressure, oxygen

_, hidden_outputs = layer_forward(patient, hidden)
_, final_outputs  = layer_forward(hidden_outputs, output)

prediction = final_outputs[0]
label      = "ABNORMAL" if prediction >= 0.5 else "NORMAL"

print(f"\nPatient reading  : {patient}")
print(f"Hidden outputs   : {[round(o, 4) for o in hidden_outputs]}")
print(f"Final prediction : {round(prediction, 4)}")
print(f"Network says     : {label}")
print(f"\nWeights are random so this prediction means nothing yet.")
print(f"Training in Phase 4 is what makes the prediction trustworthy.")
print("=" * 52)