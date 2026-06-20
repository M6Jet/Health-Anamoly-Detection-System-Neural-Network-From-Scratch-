# PHASE 5: EVALUATION AND PREDICTION INTERFACE


def save_network(network, minimums, maximums, filepath="trained_network.json"):
    """
    CONTRACT:
    IN:  network dict, minimums list, maximums list, filepath string
    OUT: None (writes file to disk)
    GUARANTEE: saves weights, biases, architecture, and normalization
               parameters needed to make predictions on new data
    
    We save minimums and maximums alongside the weights because
    new patient data must be normalized using the same scale
    as the training data. Without these values new predictions
    would be on a different scale and would be meaningless.
    """


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
    print(f"File contains weights for {network['total_params']} parameters")


def load_network(filepath="trained_network.json"):
    """
    CONTRACT:
    IN:  filepath string
    OUT: tuple (network dict, minimums list, maximums list)
    GUARANTEE: loaded network is ready to make predictions immediately
               no retraining required
    """

    with open(filepath, "r") as f:
        save_data = json.load(f)

    arch = save_data["architecture"]

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
    print(f"Architecture: {arch['num_inputs']} → "
          f"{arch['num_hidden']} → {arch['num_outputs']}")

    return network, minimums, maximums


def evaluate_at_threshold(network, test_features, test_targets, threshold=0.5):
    """
    CONTRACT:
    IN:  network, features, targets, threshold float (default 0.5)
    OUT: dictionary of evaluation metrics at this threshold
    
    Threshold controls how confident the network must be before
    flagging a patient as abnormal. Lower threshold = more sensitive
    but more false alarms. Higher threshold = fewer false alarms
    but more missed cases.
    """

    tp = fp = tn = fn = 0
    all_predictions = []

    for features, target in zip(test_features, test_targets):
        cache      = forward_pass(network, features)
        prediction = cache["prediction"]
        predicted  = 1 if prediction >= threshold else 0

        all_predictions.append(prediction)

        if predicted == 1 and target == 1:
            tp += 1
        elif predicted == 1 and target == 0:
            fp += 1
        elif predicted == 0 and target == 0:
            tn += 1
        elif predicted == 0 and target == 1:
            fn += 1

    total     = tp + fp + tn + fn
    accuracy  = (tp + tn) / total * 100     if total > 0             else 0
    precision = tp / (tp + fp) * 100        if (tp + fp) > 0         else 0
    recall    = tp / (tp + fn) * 100        if (tp + fn) > 0         else 0
    f1        = (2 * precision * recall /
                 (precision + recall))       if (precision + recall) > 0 else 0

    return {
        "threshold"        : threshold,
        "accuracy"         : round(accuracy, 2),
        "precision"        : round(precision, 2),
        "recall"           : round(recall, 2),
        "f1_score"         : round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "all_predictions"  : all_predictions
    }


def threshold_analysis(network, test_features, test_targets):
    """
    Tests the network at multiple decision thresholds and prints
    the full tradeoff table. Shows how precision and recall trade
    against each other as the threshold changes.
    
    This is a simplified version of ROC curve analysis.
    """

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"\n{'─' * 62}")
    print(f"  THRESHOLD ANALYSIS")
    print(f"{'─' * 62}")
    print(f"  {'Threshold':>10} | {'Accuracy':>9} | "
          f"{'Precision':>10} | {'Recall':>7} | {'F1':>7} | {'FN':>4}")
    print(f"  {'─'*10}-+-{'─'*9}-+-{'─'*10}-+-{'─'*7}-+-{'─'*7}-+-{'─'*4}")

    best_f1        = 0
    best_threshold = 0.5

    for t in thresholds:
        metrics = evaluate_at_threshold(network, test_features, test_targets, t)

        if metrics["f1_score"] > best_f1:
            best_f1        = metrics["f1_score"]
            best_threshold = t

        marker = " ←" if t == best_threshold else ""
        print(f"  {t:>10.1f} | {metrics['accuracy']:>8.2f}% | "
              f"{metrics['precision']:>9.2f}% | "
              f"{metrics['recall']:>6.2f}% | "
              f"{metrics['f1_score']:>6.2f}% | "
              f"{metrics['fn']:>4}{marker}")

    print(f"{'─' * 62}")
    print(f"  ← marks the threshold with the best F1 score")
    print(f"  Lower threshold = higher recall (catch more sick patients)")
    print(f"  Higher threshold = higher precision (fewer false alarms)")

    return best_threshold


def confidence_distribution(network, test_features, test_targets):
    """
    Shows how confident the network is across all predictions.
    Splits predictions into confidence buckets and shows accuracy in each.
    A well-trained network should be correct more often when it is confident.
    """

    print(f"\n{'─' * 52}")
    print(f"  CONFIDENCE DISTRIBUTION")
    print(f"{'─' * 52}")

    buckets = {
        "Very confident NORMAL (0.0-0.2)"   : [],
        "Confident NORMAL (0.2-0.4)"        : [],
        "Uncertain (0.4-0.6)"               : [],
        "Confident ABNORMAL (0.6-0.8)"      : [],
        "Very confident ABNORMAL (0.8-1.0)" : []
    }

    for features, target in zip(test_features, test_targets):
        cache = forward_pass(network, features)
        pred  = cache["prediction"]

        entry = (pred, target)

        if pred < 0.2:
            buckets["Very confident NORMAL (0.0-0.2)"].append(entry)
        elif pred < 0.4:
            buckets["Confident NORMAL (0.2-0.4)"].append(entry)
        elif pred < 0.6:
            buckets["Uncertain (0.4-0.6)"].append(entry)
        elif pred < 0.8:
            buckets["Confident ABNORMAL (0.6-0.8)"].append(entry)
        else:
            buckets["Very confident ABNORMAL (0.8-1.0)"].append(entry)

    for bucket_name, entries in buckets.items():
        if len(entries) == 0:
            print(f"  {bucket_name}: no patients")
            continue

        correct = sum(1 for pred, target in entries
                      if (pred >= 0.5) == (target == 1))
        accuracy = correct / len(entries) * 100

        bar = "█" * len(entries)
        print(f"  {bucket_name}")
        print(f"    {len(entries):>3} patients | accuracy: {round(accuracy,1):>5}% | {bar}")

    print(f"{'─' * 52}")


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


def normalize_single_patient(raw_features, minimums, maximums):
    """
    Normalizes a single patient's raw feature values using the
    same min-max scale used during training.
    
    This is critical. If training data was normalized with a certain
    scale and new patient data uses a different scale, predictions
    are meaningless. Always normalize new data with training parameters.
    """

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
    """
    Estimates how much each input feature influenced the prediction
    by running a simple sensitivity analysis.
    
    For each feature: temporarily set it to 0 and see how much
    the prediction changes. Large change = high influence.
    This is a simplified version of feature importance analysis
    used in production ML systems.
    """

    baseline_cache = forward_pass(network, normalized_inputs)
    baseline_pred  = baseline_cache["prediction"]

    influences = []

    for i in range(len(normalized_inputs)):

        modified = normalized_inputs.copy()
        modified[i] = 0.0

        modified_cache = forward_pass(network, modified)
        modified_pred  = modified_cache["prediction"]

        influence = abs(baseline_pred - modified_pred)
        influences.append((FEATURE_NAMES[i], influence))

    influences_ranked = sorted(influences, key=lambda x: x[1], reverse=True)

    return influences_ranked


def predict_patient(network, minimums, maximums, raw_features, patient_name="Patient"):
    """
    CONTRACT:
    IN:  network dict, minimums list, maximums list
         raw_features list of 13 raw values (not normalized)
         patient_name string for display
    OUT: dictionary with prediction, label, risk_level, confidence,
         top_influences
    GUARANTEE: raw_features are normalized internally before prediction
               network is not modified
    
    This is the main interface function. A doctor or nurse would
    call this with a patient's raw clinical measurements and receive
    a complete prediction report.
    """

    normalized = normalize_single_patient(raw_features, minimums, maximums)

    cache      = forward_pass(network, normalized)
    prediction = cache["prediction"]
    label      = cache["label"]

    if prediction >= 0.8:
        risk_level  = "HIGH RISK"
        risk_symbol = "🔴"
    elif prediction >= 0.6:
        risk_level  = "ELEVATED RISK"
        risk_symbol = "🟠"
    elif prediction >= 0.4:
        risk_level  = "BORDERLINE"
        risk_symbol = "🟡"
    elif prediction >= 0.2:
        risk_level  = "LOW RISK"
        risk_symbol = "🟢"
    else:
        risk_level  = "VERY LOW RISK"
        risk_symbol = "🟢"

    confidence = abs(prediction - 0.5) * 2 * 100

    top_influences = feature_influence_analysis(network, normalized)

    result = {
        "patient_name"   : patient_name,
        "prediction"     : prediction,
        "label"          : label,
        "risk_level"     : risk_level,
        "risk_symbol"    : risk_symbol,
        "confidence"     : round(confidence, 1),
        "top_influences" : top_influences
    }

    return result


def print_prediction_report(result):
    """Prints a complete formatted prediction report for one patient."""

    w = 52
    print(f"\n{'=' * w}")
    print(f"  PATIENT PREDICTION REPORT")
    print(f"{'=' * w}")
    print(f"  Patient      : {result['patient_name']}")
    print(f"  Prediction   : {round(result['prediction'], 4)}")
    print(f"  Assessment   : {result['label']}")
    print(f"  Risk Level   : {result['risk_level']} {result['risk_symbol']}")
    print(f"  Confidence   : {result['confidence']}%")
    print(f"\n  Top Features Driving This Prediction:")
    print(f"  {'─' * (w-4)}")

    for i, (feature_name, influence) in enumerate(result["top_influences"][:5]):
        bar   = "█" * int(influence * 100)
        print(f"  {i+1}. {feature_name}")
        print(f"     Influence: {round(influence, 4):.4f}  {bar}")

    print(f"{'=' * w}")
    print(f"  ⚠ This tool assists clinical decision-making.")
    print(f"  ⚠ It does not replace professional medical judgment.")
    print(f"{'=' * w}")


print("\n" + "=" * 52)
print(" PHASE 5 — EVALUATION AND PREDICTION INTERFACE")
print("=" * 52)

if os.path.exists("processed.cleveland.data"):

    save_network(net, minimums, maximums)

    print(f"\n--- Deep Evaluation ---")
    best_threshold = threshold_analysis(net, test_feat, test_targ)
    confidence_distribution(net, test_feat, test_targ)

    print(f"\n--- Prediction Interface Demo ---")


    patient_a_raw = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]


    patient_b_raw = [45, 0, 0, 120, 180, 0, 0, 170, 0, 0.0, 2, 0, 3]

    patient_c_raw = [54, 1, 1, 130, 256, 0, 0, 147, 0, 1.4, 1, 1, 2]

    for patient_raw, name in [
        (patient_a_raw, "Patient A (High Risk Profile)"),
        (patient_b_raw, "Patient B (Low Risk Profile)"),
        (patient_c_raw, "Patient C (Borderline Profile)")
    ]:
        result = predict_patient(net, minimums, maximums, patient_raw, name)
        print_prediction_report(result)

    print(f"\n--- Reload Test ---")
    loaded_net, loaded_mins, loaded_maxs = load_network()

    reload_result = predict_patient(
        loaded_net, loaded_mins, loaded_maxs,
        patient_a_raw, "Patient A (from reloaded network)"
    )

    print(f"Original prediction  : {round(result['prediction'], 6)}")
    print(f"Reloaded prediction  : {round(reload_result['prediction'], 6)}")
    print(f"Match: {'YES ✓' if round(result['prediction'], 6) == round(reload_result['prediction'], 6) else 'NO ✗'}")

print("=" * 52)

