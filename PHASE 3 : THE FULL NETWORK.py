# PHASE 3: THE FULL NETWORK


def build_network(num_inputs, num_hidden, num_outputs):
    """
    CONTRACT:
    IN:  num_inputs (int)  — number of input features per patient
         num_hidden (int)  — number of neurons in the hidden layer
         num_outputs (int) — number of output neurons (1 for binary classification)
    OUT: network dictionary containing all layers and architecture settings
    GUARANTEE: all weights initialized randomly, seed must be set before calling
    
    Builds a feedforward neural network with one hidden layer.
    Architecture: input → hidden layer → output layer → prediction
    """


    hidden_layer = create_layer(num_neurons=num_hidden, num_inputs=num_inputs)

    output_layer = create_layer(num_neurons=num_outputs, num_inputs=num_hidden)

    network = {
        "hidden_layer" : hidden_layer,
        "output_layer" : output_layer,
        "num_inputs"   : num_inputs,
        "num_hidden"   : num_hidden,
        "num_outputs"  : num_outputs
    }

    hidden_params = num_hidden * (num_inputs + 1)
    output_params = num_outputs * (num_hidden + 1)

    network["total_params"] = hidden_params + output_params

    return network


print("\n--- Network Build Test ---")

random.seed(42)

net = build_network(num_inputs=13, num_hidden=8, num_outputs=1)

print(f"Network built successfully")
print(f"Input features  : {net['num_inputs']}")
print(f"Hidden neurons  : {net['num_hidden']}")
print(f"Output neurons  : {net['num_outputs']}")
print(f"Total parameters: {net['total_params']}")
print(f"  ({net['num_hidden']} hidden neurons × ({net['num_inputs']} weights + 1 bias))")
print(f"  + ({net['num_outputs']} output neuron × ({net['num_hidden']} weights + 1 bias))")

def forward_pass(network, inputs):
    """
    CONTRACT:
    IN:  network dict (from build_network)
         inputs list of floats — one per input feature, normalized 0 to 1
    OUT: cache dictionary containing:
            input         — original inputs (unchanged)
            hidden_raw    — raw weighted sums from hidden layer
            hidden_output — activated outputs from hidden layer
            output_raw    — raw weighted sums from output layer
            output_output — final activated prediction(s)
            prediction    — single float, the network's confidence (0 to 1)
            label         — string, ABNORMAL if prediction >= 0.5 else NORMAL
    GUARANTEE: network is not modified. inputs are not modified.
    """

    if len(inputs) != network["num_inputs"]:
        print(f"ERROR: expected {network['num_inputs']} inputs, got {len(inputs)}")
        return None

    hidden_raw, hidden_output = layer_forward(inputs, network["hidden_layer"])

    output_raw, output_output = layer_forward(hidden_output, network["output_layer"])

    cache = {
        "input"         : inputs,
        "hidden_raw"    : hidden_raw,
        "hidden_output" : hidden_output,
        "output_raw"    : output_raw,
        "output_output" : output_output,
        "prediction"    : output_output[0],
        "label"         : "ABNORMAL" if output_output[0] >= 0.5 else "NORMAL"
    }

    return cache


print("\n--- Forward Pass Test ---")

test_patient = [0.71, 1.0, 0.67, 0.49, 0.54,
                0.0,  0.5, 0.73, 0.0,  0.35,
                0.67, 0.0, 0.67]

cache = forward_pass(net, test_patient)

print(f"Input features   : {len(cache['input'])} values")
print(f"Hidden outputs   : {[round(o, 4) for o in cache['hidden_output']]}")
print(f"Final prediction : {round(cache['prediction'], 4)}")
print(f"Network says     : {cache['label']}")

def forward_pass_verbose(network, inputs, patient_id=1):
    """
    Identical to forward_pass but prints every intermediate calculation.
    Use this to trace exactly how data flows through the network.
    Not used during training — only for learning and debugging.
    
    Parameters:
        network:    network dictionary
        inputs:     list of input features
        patient_id: label for the print output (int, default 1)
    """

    print(f"\n{'─' * 52}")
    print(f"  VERBOSE FORWARD PASS — Patient {patient_id}")
    print(f"{'─' * 52}")
    print(f"  Raw inputs: {[round(x, 3) for x in inputs]}")

    print(f"\n  HIDDEN LAYER ({network['num_hidden']} neurons)")

    hidden_raw    = []
    hidden_output = []

    for i, neuron in enumerate(network["hidden_layer"]):

        
        weighted_sum = sum(inp * w for inp, w in zip(inputs, neuron["weights"]))
        raw          = weighted_sum + neuron["bias"]
        out          = sigmoid(raw)
        fires        = "FIRES" if out >= 0.5 else "quiet"

        hidden_raw.append(raw)
        hidden_output.append(out)

        print(f"  Neuron {i+1}: weighted_sum={round(weighted_sum,4):>7} "
              f"+ bias={round(neuron['bias'],4):>7} "
              f"= raw={round(raw,4):>7} "
              f"→ sigmoid={round(out,4):.4f} ({fires})")

    print(f"\n  OUTPUT LAYER (1 neuron)")

    output_neuron = network["output_layer"][0]
    weighted_sum  = sum(h * w for h, w in zip(hidden_output,
                                               output_neuron["weights"]))
    raw           = weighted_sum + output_neuron["bias"]
    prediction    = sigmoid(raw)
    label         = "ABNORMAL" if prediction >= 0.5 else "NORMAL"

    print(f"  Neuron 1: weighted_sum={round(weighted_sum,4):>7} "
          f"+ bias={round(output_neuron['bias'],4):>7} "
          f"= raw={round(raw,4):>7} "
          f"→ sigmoid={round(prediction,4):.4f}")

    print(f"\n  {'─' * 48}")
    print(f"  PREDICTION : {round(prediction, 4)} → {label}")
    print(f"  {'─' * 48}")

    return forward_pass(network, inputs)

def batch_forward_pass(network, batch_inputs):
    """
    CONTRACT:
    IN:  network dict, batch_inputs list of lists
         (each inner list is one patient's features)
    OUT: list of cache dictionaries, one per patient
    GUARANTEE: processes patients independently, order preserved
    """

    results = [forward_pass(network, patient) for patient in batch_inputs]

    return results


def batch_predictions_summary(results, targets=None):
    """
    Prints a summary of predictions across a batch.
    If targets are provided, also shows accuracy.
    
    Parameters:
        results: list of cache dictionaries from batch_forward_pass
        targets: list of correct labels (optional, 0 or 1 per patient)
    """

    print(f"\n{'─' * 52}")
    print(f"  BATCH PREDICTION SUMMARY")
    print(f"{'─' * 52}")

    correct = 0

    for i, cache in enumerate(results):
        pred  = cache["prediction"]
        label = cache["label"]

        if targets is not None:
            true_label = "ABNORMAL" if targets[i] == 1 else "NORMAL"
            match      = "✓" if label == true_label else "✗"
            if label == true_label:
                correct += 1
            print(f"  Patient {i+1:02d}: {round(pred, 4):.4f} → {label:<10} "
                  f"| Actual: {true_label:<10} {match}")
        else:
            print(f"  Patient {i+1:02d}: {round(pred, 4):.4f} → {label}")

    if targets is not None:
        accuracy = correct / len(results) * 100
        print(f"\n  Accuracy: {correct}/{len(results)} = {round(accuracy, 1)}%")
        print(f"  (Random weights — accuracy means nothing yet)")

    print(f"{'─' * 52}")



print("\n" + "=" * 52)
print(" PHASE 3 SUMMARY — FULL NETWORK FORWARD PASS")
print("=" * 52)

cache = forward_pass_verbose(net, test_patient, patient_id=1)

batch = [
    [0.71, 1.0, 0.67, 0.49, 0.54, 0.0, 0.5, 0.73, 0.0, 0.35, 0.67, 0.0, 0.67],
    [0.44, 0.0, 0.33, 0.61, 0.42, 1.0, 0.0, 0.81, 0.0, 0.0,  0.33, 0.0, 0.33],
    [0.85, 1.0, 1.0,  0.73, 0.68, 0.0, 1.0, 0.44, 1.0, 0.65, 1.0,  0.67, 1.0],
    [0.33, 0.0, 0.0,  0.37, 0.31, 0.0, 0.5, 0.93, 0.0, 0.0,  0.33, 0.0, 0.33],
    [0.63, 1.0, 0.67, 0.55, 0.58, 0.0, 0.5, 0.61, 1.0, 0.45, 0.67, 0.33, 0.67],
    [0.52, 0.0, 0.33, 0.45, 0.39, 0.0, 0.0, 0.88, 0.0, 0.1,  0.33, 0.0, 0.33],
]

targets = [1, 0, 1, 0, 1, 0]

results = batch_forward_pass(net, batch)
batch_predictions_summary(results, targets)

print(f"\nNetwork architecture confirmed working end to end.")
print(f"13 features in → prediction out in a single function call.")
print(f"Accuracy with random weights is meaningless.")
print(f"Phase 4 training is what makes these predictions real.")
print("=" * 52)