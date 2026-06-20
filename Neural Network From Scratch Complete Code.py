# ============================================================
# NEURAL NETWORK FROM SCRATCH
# Health Anomaly Detection System
# UCI Heart Disease Dataset — Binary Classification
# Deployment: AWS Lambda
# ============================================================

import math
import random
import csv
import os
import json


# ============================================================
# CONFIGURATION
# ============================================================

random.seed(42)


# ============================================================
# PHASE 1 — CORE MATH FUNCTIONS
# ============================================================

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def mean_squared_error(predictions, targets):
    if len(predictions) != len(targets):
        print("ERROR: predictions and targets must be the same length")
        return None
    n = len(predictions)
    squared_errors = [(predictions[i] - targets[i]) ** 2 for i in range(n)]
    return sum(squared_errors) / n


def mean_squared_error_derivative(predictions, targets):
    n = len(predictions)
    return [2 * (predictions[i] - targets[i]) / n for i in range(n)]


# ============================================================
# PHASE 2 — NEURON AND LAYER
# ============================================================

def create_neuron(num_inputs):
    return {
        "weights": [random.uniform(-1, 1) for _ in range(num_inputs)],
        "bias"   : random.uniform(-1, 1)
    }


def neuron_forward(inputs, neuron, activation_function=sigmoid):
    weighted_sum = sum(inp * w for inp, w in zip(inputs, neuron["weights"]))
    raw          = weighted_sum + neuron["bias"]
    output       = activation_function(raw)
    return raw, output


def create_layer(num_neurons, num_inputs):
    return [create_neuron(num_inputs) for _ in range(num_neurons)]


def layer_forward(inputs, layer, activation_function=sigmoid):
    raw_sums = []
    outputs  = []
    for neuron in layer:
        raw, out = neuron_forward(inputs, neuron, activation_function)
        raw_sums.append(raw)
        outputs.append(out)
    return raw_sums, outputs


# ============================================================
# PHASE 3 — FULL NETWORK
# ============================================================

def build_network(num_inputs, num_hidden, num_outputs):
    hidden_layer  = create_layer(num_neurons=num_hidden, num_inputs=num_inputs)
    output_layer  = create_layer(num_neurons=num_outputs, num_inputs=num_hidden)
    hidden_params = num_hidden * (num_inputs + 1)
    output_params = num_outputs * (num_hidden + 1)

    return {
        "hidden_layer" : hidden_layer,
        "output_layer" : output_layer,
        "num_inputs"   : num_inputs,
        "num_hidden"   : num_hidden,
        "num_outputs"  : num_outputs,
        "total_params" : hidden_params + output_params
    }


def forward_pass(network, inputs):
    if len(inputs) != network["num_inputs"]:
        print(f"ERROR: expected {network['num_inputs']} inputs, got {len(inputs)}")
        return None

    hidden_raw, hidden_output = layer_forward(inputs, network["hidden_layer"])
    output_raw, output_output = layer_forward(hidden_output, network["output_layer"])

    return {
        "input"         : inputs,
        "hidden_raw"    : hidden_raw,
        "hidden_output" : hidden_output,
        "output_raw"    : output_raw,
        "output_output" : output_output,
        "prediction"    : output_output[0],
        "label"         : "ABNORMAL" if output_output[0] >= 0.5 else "NORMAL"
    }


def batch_forward_pass(network, batch_inputs):
    return [forward_pass(network, patient) for patient in batch_inputs]


# ============================================================
# PHASE 4 — DATA LOADING AND TRAINING
# ============================================================

def load_heart_data(filepath):
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
                target   = 1 if target > 0 else 0
                features_list.append(features)
                targets_list.append(target)
            except ValueError:
                skipped += 1
                continue

    print(f"Data loaded: {len(features_list)} patients ({skipped} rows skipped)")
    print(f"Abnormal: {sum(targets_list)} | Normal: {len(targets_list) - sum(targets_list)}")
    return features_list, targets_list


def normalize_features(features_list):
    num_features = len(features_list[0])
    minimums     = []
    maximums     = []

    for feature_index in range(num_features):
        column_values = [patient[feature_index] for patient in features_list]
        minimums.append(min(column_values))
        maximums.append(max(column_values))

    normalized_list = []
    for patient in features_list:
        normalized_patient = []
        for i in range(num_features):
            min_val = minimums[i]
            max_val = maximums[i]
            val     = patient[i]
            if max_val == min_val:
                normalized_patient.append(0.0)
            else:
                normalized_patient.append((val - min_val) / (max_val - min_val))
        normalized_list.append(normalized_patient)

    return normalized_list, minimums, maximums


def train_test_split(features, targets, test_ratio=0.2, seed=42):
    random.seed(seed)
    indices     = list(range(len(features)))
    random.shuffle(indices)
    split_point = int(len(indices) * (1 - test_ratio))
    train_idx   = indices[:split_point]
    test_idx    = indices[split_point:]

    return (
        [features[i] for i in train_idx],
        [features[i] for i in test_idx],
        [targets[i]  for i in train_idx],
        [targets[i]  for i in test_idx]
    )


def backward_pass(network, cache, target, learning_rate=0.1):
    prediction    = cache["prediction"]
    output_raw    = cache["output_raw"][0]
    hidden_output = cache["hidden_output"]
    hidden_raw    = cache["hidden_raw"]
    inputs        = cache["input"]

    loss         = (prediction - target) ** 2
    loss_grad    = 2 * (prediction - target)
    output_delta = loss_grad * sigmoid_derivative(sigmoid(output_raw))

    output_neuron = network["output_layer"][0]

    for i, hidden_out in enumerate(hidden_output):
        output_neuron["weights"][i] -= learning_rate * (output_delta * hidden_out)
    output_neuron["bias"] -= learning_rate * output_delta

    hidden_deltas = []
    for j, hidden_neuron in enumerate(network["hidden_layer"]):
        error_signal  = output_delta * output_neuron["weights"][j]
        hidden_delta  = error_signal * sigmoid_derivative(sigmoid(hidden_raw[j]))
        hidden_deltas.append(hidden_delta)

    for j, hidden_neuron in enumerate(network["hidden_layer"]):
        for i, input_val in enumerate(inputs):
            hidden_neuron["weights"][i] -= learning_rate * (hidden_deltas[j] * input_val)
        hidden_neuron["bias"] -= learning_rate * hidden_deltas[j]

    return loss


def train_network(network, train_features, train_targets,
                  epochs=200, learning_rate=0.1, print_every=20):
    loss_history = []

    print(f"\n--- Training Started ---")
    print(f"Patients   : {len(train_features)}")
    print(f"Epochs     : {epochs}")
    print(f"Learn rate : {learning_rate}")
    print(f"{'─' * 42}")

    for epoch in range(epochs):
        epoch_loss = 0
        correct    = 0

        combined  = list(zip(train_features, train_targets))
        random.shuffle(combined)
        feats_shuffled, targs_shuffled = zip(*combined)

        for features, target in zip(feats_shuffled, targs_shuffled):
            cache = forward_pass(network, list(features))
            if cache is None:
                continue
            loss       = backward_pass(network, cache, target, learning_rate)
            epoch_loss += loss
            predicted  = 1 if cache["prediction"] >= 0.5 else 0
            if predicted == target:
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


# ============================================================
# PHASE 5 — EVALUATION AND PREDICTION INTERFACE
# ============================================================

FEATURE_NAMES = [
    "Age",
    "Sex (1=male, 0=female)",
    "Chest pain type (0-3)",
    "Resting blood pressure",
    "Cholesterol",
    "Fasting blood sugar > 120 (1=yes, 0=no)",
    "Resting ECG result (0-2)",
    "Max heart rate achieved",
    "Exercise induced angina (1=yes, 0=no)",
    "ST depression",
    "ST slope (0-2)",
    "Number of major vessels (0-3)",
    "Thal (1=normal, 2=fixed defect, 3=reversible defect)"
]


def evaluate_network(network, test_features, test_targets):
    tp = fp = tn = fn = 0

    for features, target in zip(test_features, test_targets):
        cache     = forward_pass(network, features)
        predicted = 1 if cache["prediction"] >= 0.5 else 0

        if   predicted == 1 and target == 1: tp += 1
        elif predicted == 1 and target == 0: fp += 1
        elif predicted == 0 and target == 0: tn += 1
        elif predicted == 0 and target == 1: fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total * 100          if total > 0             else 0
    precision = tp / (tp + fp) * 100             if (tp + fp) > 0         else 0
    recall    = tp / (tp + fn) * 100             if (tp + fn) > 0         else 0
    f1        = (2 * precision * recall /
                 (precision + recall))            if (precision + recall) > 0 else 0

    return {
        "accuracy"  : round(accuracy, 2),
        "precision" : round(precision, 2),
        "recall"    : round(recall, 2),
        "f1_score"  : round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total"     : total
    }


def print_evaluation(results):
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
    print(f"\n  False Negatives : {results['fn']} patients with disease missed")
    print(f"  False Positives : {results['fp']} healthy patients wrongly flagged")
    print(f"{'=' * 52}")


def evaluate_at_threshold(network, test_features, test_targets, threshold=0.5):
    tp = fp = tn = fn = 0

    for features, target in zip(test_features, test_targets):
        cache     = forward_pass(network, features)
        predicted = 1 if cache["prediction"] >= threshold else 0

        if   predicted == 1 and target == 1: tp += 1
        elif predicted == 1 and target == 0: fp += 1
        elif predicted == 0 and target == 0: tn += 1
        elif predicted == 0 and target == 1: fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total * 100          if total > 0             else 0
    precision = tp / (tp + fp) * 100             if (tp + fp) > 0         else 0
    recall    = tp / (tp + fn) * 100             if (tp + fn) > 0         else 0
    f1        = (2 * precision * recall /
                 (precision + recall))            if (precision + recall) > 0 else 0

    return {
        "threshold" : threshold,
        "accuracy"  : round(accuracy, 2),
        "precision" : round(precision, 2),
        "recall"    : round(recall, 2),
        "f1_score"  : round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def threshold_analysis(network, test_features, test_targets):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1    = 0
    best_t     = 0.5
    results    = []

    for t in thresholds:
        m = evaluate_at_threshold(network, test_features, test_targets, t)
        results.append(m)
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_t  = t

    print(f"\n{'─' * 62}")
    print(f"  THRESHOLD ANALYSIS")
    print(f"{'─' * 62}")
    print(f"  {'Threshold':>10} | {'Accuracy':>9} | "
          f"{'Precision':>10} | {'Recall':>7} | {'F1':>7} | {'FN':>4}")
    print(f"  {'─'*10}-+-{'─'*9}-+-{'─'*10}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}")

    for m in results:
        marker = " ←" if m["threshold"] == best_t else ""
        print(f"  {m['threshold']:>10.1f} | {m['accuracy']:>8.2f}% | "
              f"{m['precision']:>9.2f}% | "
              f"{m['recall']:>6.2f}% | "
              f"{m['f1_score']:>6.2f}% | "
              f"{m['fn']:>4}{marker}")

    print(f"{'─' * 62}")
    return best_t


def save_network(network, minimums, maximums, filepath="trained_network.json"):
    save_data = {
        "architecture" : {
            "num_inputs"  : network["num_inputs"],
            "num_hidden"  : network["num_hidden"],
            "num_outputs" : network["num_outputs"]
        },
        "hidden_layer" : network["hidden_layer"],
        "output_layer" : network["output_layer"],
        "normalization": {
            "minimums" : minimums,
            "maximums" : maximums
        }
    }
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nNetwork saved to {filepath}")
    print(f"Total parameters saved: {network['total_params']}")


def load_network(filepath="trained_network.json"):
    with open(filepath, "r") as f:
        save_data = json.load(f)
    arch    = save_data["architecture"]
    network = {
        "hidden_layer" : save_data["hidden_layer"],
        "output_layer" : save_data["output_layer"],
        "num_inputs"   : arch["num_inputs"],
        "num_hidden"   : arch["num_hidden"],
        "num_outputs"  : arch["num_outputs"],
        "total_params" : (arch["num_hidden"] * (arch["num_inputs"] + 1) +
                          arch["num_outputs"] * (arch["num_hidden"] + 1))
    }
    minimums = save_data["normalization"]["minimums"]
    maximums = save_data["normalization"]["maximums"]
    print(f"Network loaded from {filepath}")
    return network, minimums, maximums


def normalize_single_patient(raw_features, minimums, maximums):
    normalized = []
    for i, value in enumerate(raw_features):
        min_val = minimums[i]
        max_val = maximums[i]
        if max_val == min_val:
            normalized.append(0.0)
        else:
            normalized.append((value - min_val) / (max_val - min_val))
    return normalized


def feature_influence_analysis(network, normalized_inputs):
    baseline_cache = forward_pass(network, normalized_inputs)
    baseline_pred  = baseline_cache["prediction"]
    influences     = []

    for i in range(len(normalized_inputs)):
        modified    = normalized_inputs.copy()
        modified[i] = 0.0
        mod_cache   = forward_pass(network, modified)
        influence   = abs(baseline_pred - mod_cache["prediction"])
        influences.append((FEATURE_NAMES[i], influence))

    return sorted(influences, key=lambda x: x[1], reverse=True)


def predict_patient(network, minimums, maximums, raw_features, patient_name="Patient"):
    normalized = normalize_single_patient(raw_features, minimums, maximums)
    cache      = forward_pass(network, normalized)
    prediction = cache["prediction"]
    label      = cache["label"]

    if prediction >= 0.8:
        risk_level = "HIGH RISK"
    elif prediction >= 0.6:
        risk_level = "ELEVATED RISK"
    elif prediction >= 0.4:
        risk_level = "BORDERLINE"
    elif prediction >= 0.2:
        risk_level = "LOW RISK"
    else:
        risk_level = "VERY LOW RISK"

    confidence     = round(abs(prediction - 0.5) * 2 * 100, 1)
    top_influences = feature_influence_analysis(network, normalized)

    return {
        "patient_name"   : patient_name,
        "prediction"     : prediction,
        "label"          : label,
        "risk_level"     : risk_level,
        "confidence"     : confidence,
        "top_influences" : top_influences
    }


def print_prediction_report(result):
    w = 52
    print(f"\n{'=' * w}")
    print(f"  PATIENT PREDICTION REPORT")
    print(f"{'=' * w}")
    print(f"  Patient    : {result['patient_name']}")
    print(f"  Prediction : {round(result['prediction'], 4)}")
    print(f"  Assessment : {result['label']}")
    print(f"  Risk Level : {result['risk_level']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"\n  Top Features Driving This Prediction:")
    print(f"  {'─' * (w - 4)}")
    for i, (name, influence) in enumerate(result["top_influences"][:5]):
        bar = "█" * int(influence * 100)
        print(f"  {i+1}. {name}")
        print(f"     Influence: {round(influence, 4):.4f}  {bar}")
    print(f"{'=' * w}")
    print(f"  This tool assists clinical decision-making.")
    print(f"  It does not replace professional medical judgment.")
    print(f"{'=' * w}")


# ============================================================
# MAIN — FULL PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 52)
    print(" NEURAL NETWORK — HEALTH ANOMALY DETECTION")
    print("=" * 52)

    filepath = "processed.cleveland.data"

    if not os.path.exists(filepath):
        print(f"\nERROR: Could not find {filepath}")
        print(f"Download from archive.ics.uci.edu/dataset/45/heart+disease")
        print(f"Save in the same folder as this script")

    else:

        # --- LOAD AND PREPARE DATA ---
        raw_features, targets             = load_heart_data(filepath)
        features, minimums, maximums      = normalize_features(raw_features)
        train_feat, test_feat, train_targ, test_targ = train_test_split(
            features, targets, test_ratio=0.2
        )

        print(f"\nTraining patients : {len(train_feat)}")
        print(f"Test patients     : {len(test_feat)}")

        # --- BUILD NETWORK ---
        random.seed(42)
        net = build_network(num_inputs=13, num_hidden=8, num_outputs=1)

        print(f"\nNetwork  : 13 inputs → 8 hidden → 1 output")
        print(f"Parameters: {net['total_params']}")

        # --- BEFORE TRAINING ---
        before = evaluate_network(net, test_feat, test_targ)
        print(f"\nAccuracy before training: {before['accuracy']}%")

        # --- TRAIN ---
        loss_history = train_network(
            network        = net,
            train_features = train_feat,
            train_targets  = train_targ,
            epochs         = 200,
            learning_rate  = 0.1,
            print_every    = 20
        )

        # --- EVALUATE ---
        after = evaluate_network(net, test_feat, test_targ)
        print_evaluation(after)

        improvement = after["accuracy"] - before["accuracy"]
        print(f"\n  Accuracy improvement : +{round(improvement, 2)}%")
        print(f"  Loss at epoch 1      : {round(loss_history[0], 6)}")
        print(f"  Loss at epoch 200    : {round(loss_history[-1], 6)}")
        print(f"  Loss reduction       : {round(loss_history[0] - loss_history[-1], 6)}")

        # --- THRESHOLD ANALYSIS ---
        best_threshold = threshold_analysis(net, test_feat, test_targ)

        # --- SAVE ---
        save_network(net, minimums, maximums)

        # --- PREDICTION INTERFACE DEMO ---
        print(f"\n--- Prediction Interface Demo ---")

        patient_a = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        patient_b = [45, 0, 0, 120, 180, 0, 0, 170, 0, 0.0, 2, 0, 3]
        patient_c = [54, 1, 1, 130, 256, 0, 0, 147, 0, 1.4, 1, 1, 2]

        for raw, name in [
            (patient_a, "Patient A — High Risk Profile"),
            (patient_b, "Patient B — Low Risk Profile"),
            (patient_c, "Patient C — Borderline Profile")
        ]:
            result = predict_patient(net, minimums, maximums, raw, name)
            print_prediction_report(result)

        # --- RELOAD TEST ---
        print(f"\n--- Reload Test ---")
        loaded_net, loaded_mins, loaded_maxs = load_network()
        reload_result = predict_patient(
            loaded_net, loaded_mins, loaded_maxs,
            patient_a, "Patient A (reloaded)"
        )
        match = round(result["prediction"], 6) == round(reload_result["prediction"], 6)
        print(f"Original prediction  : {round(result['prediction'], 6)}")
        print(f"Reloaded prediction  : {round(reload_result['prediction'], 6)}")
        print(f"Match: {'YES ✓' if match else 'NO ✗'}")

        print(f"\n{'=' * 52}")
        print(f" Pipeline complete. trained_network.json ready for Lambda.")
        print(f"{'=' * 52}")