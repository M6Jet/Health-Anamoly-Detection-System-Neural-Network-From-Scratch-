import math
import random
import csv
import os

# PHASE 4 - TRAINING THE DATA


def load_heart_data(filepath):
    """
    CONTRACT:
    IN:  filepath string — path to processed.cleveland.data
    OUT: tuple (features_list, targets_list)
         features_list: list of lists, each inner list has 13 floats
         targets_list:  list of ints, each is 0 (normal) or 1 (abnormal)
    GUARANTEE: rows with missing values (marked as ?) are skipped
               targets above 0 are converted to 1 (binary classification)
    """

    features_list = []
    targets_list  = []
    skipped       = 0


    with open(filepath, "r", newline="") as file:

        reader = csv.reader(file)

        for row in reader:

            if not row:
                continue


            if any(value.strip() == "?" for value in row):
                skipped += 1
                continue

            try:
                features = [float(row[i]) for i in range(13)]
                target   = int(float(row[13]))

                target = 1 if target > 0 else 0

                features_list.append(features)
                targets_list.append(target)

            except ValueError:
                skipped += 1
                continue

    print(f"Data loaded: {len(features_list)} patients ({skipped} rows skipped)")
    print(f"Abnormal: {sum(targets_list)} | Normal: {len(targets_list) - sum(targets_list)}")

    return features_list, targets_list


def normalize_features(features_list):
    """
    CONTRACT:
    IN:  features_list — list of lists, raw feature values
    OUT: normalized_list — same structure, values scaled to 0-1 range
    GUARANTEE: each feature column scaled independently using min-max scaling
               original features_list is not modified
    
    Why normalize: neural networks work best when all inputs are on the same scale.
    A feature ranging from 0 to 300 would dominate one ranging from 0 to 1
    without normalization. Scaling everything to 0-1 puts all features on equal footing.
    """

    num_features = len(features_list[0])

    minimums = []
    maximums = []

    for feature_index in range(num_features):

        # Extract all values for this feature across all patients
        column_values = [patient[feature_index] for patient in features_list]

        minimums.append(min(column_values))
        maximums.append(max(column_values))

    normalized_list = []

    for patient in features_list:
        normalized_patient = []

        for feature_index in range(num_features):
            min_val   = minimums[feature_index]
            max_val   = maximums[feature_index]
            raw_value = patient[feature_index]

            if max_val == min_val:
                normalized_value = 0.0
            else:
                normalized_value = (raw_value - min_val) / (max_val - min_val)

            normalized_patient.append(normalized_value)

        normalized_list.append(normalized_patient)

    return normalized_list, minimums, maximums


def train_test_split(features, targets, test_ratio=0.2, seed=42):
    """
    CONTRACT:
    IN:  features list, targets list, test_ratio float (default 0.2)
    OUT: tuple of four lists (train_features, test_features,
                               train_targets, test_targets)
    GUARANTEE: 80% of data goes to training, 20% to testing
               split is random but reproducible with seed
               original lists are not modified
    
    Why split: you cannot test a network on data it trained on.
    That would be like memorizing the answers to a test —
    high score but no real understanding. The test set is data
    the network has never seen during training. Performance on
    test data is the only honest measure of how well it learned.
    """

    random.seed(seed)

    indices = list(range(len(features)))
    random.shuffle(indices)

    split_point = int(len(indices) * (1 - test_ratio))

    train_indices = indices[:split_point]
    test_indices  = indices[split_point:]

    train_features = [features[i] for i in train_indices]
    test_features  = [features[i] for i in test_indices]
    train_targets  = [targets[i]  for i in train_indices]
    test_targets   = [targets[i]  for i in test_indices]

    return train_features, test_features, train_targets, test_targets


def backward_pass(network, cache, target, learning_rate=0.1):
    """
    CONTRACT:
    IN:  network dict (will be modified — weights updated in place)
         cache dict from forward_pass (contains all intermediate values)
         target float (0 or 1 — the correct answer for this patient)
         learning_rate float (how much to adjust weights, default 0.1)
    OUT: float — the loss for this training example
    GUARANTEE: every weight and bias in the network is updated
               cache is not modified
    
    This is backpropagation. It works backward from the output layer
    to the hidden layer, calculating how much each weight contributed
    to the error and adjusting it to reduce that error.
    """

    prediction    = cache["prediction"]
    output_raw    = cache["output_raw"][0]
    hidden_output = cache["hidden_output"]

    loss = (prediction - target) ** 2

    loss_grad    = 2 * (prediction - target)
    output_delta = loss_grad * sigmoid_derivative(sigmoid(output_raw))

    output_neuron = network["output_layer"][0]

    for i, hidden_out in enumerate(hidden_output):

        weight_gradient = output_delta * hidden_out

        output_neuron["weights"][i] -= learning_rate * weight_gradient


    output_neuron["bias"] -= learning_rate * output_delta



    hidden_raw   = cache["hidden_raw"]
    hidden_deltas = []

    for j, hidden_neuron in enumerate(network["hidden_layer"]):


        error_from_output = output_delta * output_neuron["weights"][j]

        hidden_delta = error_from_output * sigmoid_derivative(sigmoid(hidden_raw[j]))
        hidden_deltas.append(hidden_delta)

    inputs = cache["input"]

    for j, hidden_neuron in enumerate(network["hidden_layer"]):
        for i, input_val in enumerate(inputs):

            weight_gradient = hidden_deltas[j] * input_val
            hidden_neuron["weights"][i] -= learning_rate * weight_gradient

        hidden_neuron["bias"] -= learning_rate * hidden_deltas[j]

    return loss


def train_network(network, train_features, train_targets,
                  epochs=100, learning_rate=0.1, print_every=10):
    """
    CONTRACT:
    IN:  network dict (weights will be updated across all epochs)
         train_features list of lists
         train_targets  list of ints
         epochs         int — how many full passes through the data (default 100)
         learning_rate  float — step size for weight updates (default 0.1)
         print_every    int — print progress every N epochs (default 10)
    OUT: loss_history list — average loss after each epoch
    GUARANTEE: network weights are updated in place after every patient
    
    One epoch = one full pass through all training data.
    We run multiple epochs because the network improves gradually.
    Each pass refines the weights a little more.
    """

    loss_history = []

    print(f"\n--- Training Started ---")
    print(f"Patients    : {len(train_features)}")
    print(f"Epochs      : {epochs}")
    print(f"Learn rate  : {learning_rate}")
    print(f"{'─' * 42}")

    for epoch in range(epochs):

        epoch_loss   = 0
        correct      = 0

        combined    = list(zip(train_features, train_targets))
        random.shuffle(combined)
        features_shuffled, targets_shuffled = zip(*combined)

        for features, target in zip(features_shuffled, targets_shuffled):

            cache = forward_pass(network, list(features))

            if cache is None:
                continue

            loss = backward_pass(network, cache, target, learning_rate)

            epoch_loss += loss

            predicted_label = 1 if cache["prediction"] >= 0.5 else 0
            if predicted_label == target:
                correct += 1

        avg_loss = epoch_loss / len(train_features)
        accuracy = correct / len(train_features) * 100
        loss_history.append(avg_loss)

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>4}/{epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Accuracy: {round(accuracy, 1):>5}%")

    print(f"{'─' * 42}")
    print(f"Training complete.")

    return loss_history


def evaluate_network(network, test_features, test_targets):
    """
    CONTRACT:
    IN:  network dict (weights not modified)
         test_features list of lists — data network has never seen
         test_targets  list of ints
    OUT: dictionary containing accuracy, true positives, true negatives,
         false positives, false negatives, precision, recall
    GUARANTEE: network is not modified during evaluation
    
    Why these metrics matter for healthcare:
    False negative = network says NORMAL but patient is ABNORMAL
                     This is dangerous — a sick patient goes undetected
    False positive = network says ABNORMAL but patient is NORMAL
                     This causes unnecessary worry and tests
    In healthcare false negatives are worse than false positives.
    Recall measures how well we catch the abnormal cases.
    """

    tp = fp = tn = fn = 0

    for features, target in zip(test_features, test_targets):
        cache     = forward_pass(network, features)
        predicted = 1 if cache["prediction"] >= 0.5 else 0

        if predicted == 1 and target == 1:
            tp += 1    # correctly identified abnormal
        elif predicted == 1 and target == 0:
            fp += 1    # wrongly flagged as abnormal
        elif predicted == 0 and target == 0:
            tn += 1    # correctly identified normal
        elif predicted == 0 and target == 1:
            fn += 1    # missed an abnormal — most dangerous error

    total    = tp + fp + tn + fn
    accuracy  = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    results = {
        "accuracy"  : round(accuracy, 2),
        "precision" : round(precision, 2),
        "recall"    : round(recall, 2),
        "f1_score"  : round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total"     : total
    }

    return results


def print_evaluation(results):
    """Prints a formatted evaluation report."""

    print(f"\n{'=' * 52}")
    print(f"  NETWORK EVALUATION REPORT")
    print(f"{'=' * 52}")
    print(f"\n  Confusion Matrix:")
    print(f"  ┌─────────────┬────────────┬────────────┐")
    print(f"  │             │ Pred ABNORM│ Pred NORMAL│")
    print(f"  ├─────────────┼────────────┼────────────┤")
    print(f"  │ True ABNORM │ TP: {results['tp']:>5}  │ FN: {results['fn']:>5}  │")
    print(f"  │ True NORMAL │ FP: {results['fp']:>5}  │ TN: {results['tn']:>5}  │")
    print(f"  └─────────────┴────────────┴────────────┘")
    print(f"\n  Accuracy  : {results['accuracy']}%")
    print(f"  Precision : {results['precision']}%")
    print(f"  Recall    : {results['recall']}%  ← most important in healthcare")
    print(f"  F1 Score  : {results['f1_score']}%")
    print(f"\n  False Negatives: {results['fn']} patients with disease missed")
    print(f"  False Positives: {results['fp']} healthy patients wrongly flagged")
    print(f"{'=' * 52}")


# PHASE 4 SUMMARY — FULL TRAINING RUN

print("\n" + "=" * 52)
print(" PHASE 4 — FULL TRAINING PIPELINE")
print("=" * 52)

filepath = "processed.cleveland.data"

if not os.path.exists(filepath):
    print(f"\nERROR: Could not find {filepath}")
    print(f"Download it from archive.ics.uci.edu/dataset/45/heart+disease")
    print(f"Save it in the same folder as neural_network.py")
else:

    raw_features, targets = load_heart_data(filepath)

    features, minimums, maximums = normalize_features(raw_features)

    train_feat, test_feat, train_targ, test_targ = train_test_split(
        features, targets, test_ratio=0.2
    )

    print(f"\nTraining patients : {len(train_feat)}")
    print(f"Test patients     : {len(test_feat)}")

    random.seed(42)
    net = build_network(num_inputs=13, num_hidden=8, num_outputs=1)

    print(f"\nNetwork: 13 inputs → 8 hidden → 1 output")
    print(f"Total parameters: {net['total_params']}")

    print(f"\n--- Before Training ---")
    before = evaluate_network(net, test_feat, test_targ)
    print(f"Accuracy before training: {before['accuracy']}%")

    loss_history = train_network(
        network       = net,
        train_features = train_feat,
        train_targets  = train_targ,
        epochs         = 200,
        learning_rate  = 0.1,
        print_every    = 20
    )

    print(f"\n--- After Training ---")
    after = evaluate_network(net, test_feat, test_targ)
    print_evaluation(after)

    improvement = after["accuracy"] - before["accuracy"]
    print(f"\n  Accuracy improvement: +{round(improvement, 2)}%")
    print(f"  Loss at epoch 1  : {round(loss_history[0], 6)}")
    print(f"  Loss at epoch 200: {round(loss_history[-1], 6)}")
    print(f"  Loss reduction   : {round(loss_history[0] - loss_history[-1], 6)}")
    print(f"{'=' * 52}")


