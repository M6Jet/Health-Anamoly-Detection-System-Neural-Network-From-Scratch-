# Health Anomaly Detection — Technical Documentation

A feedforward neural network built from scratch in pure Python, trained on real clinical data, and deployed as a public cloud inference API. No machine learning libraries were used at any point — every neuron, layer, the forward pass, backpropagation, and gradient descent were implemented by hand.

---

## 1. Overview

This system predicts whether a patient's clinical readings indicate the presence of heart disease. It takes 13 medical features as input and returns a probability between 0 and 1, along with a risk classification and the features that most influenced the prediction.

The project is deliberately built without TensorFlow, PyTorch, scikit-learn, or NumPy. The goal was to understand neural networks at the implementation level — to know exactly what those libraries do under the hood by building the same machinery from first principles using only Python's standard library.

**Live endpoint:** `POST https://jhqy8ublrj.execute-api.us-east-1.amazonaws.com/predict`

---

## 2. Architecture

The system has two distinct halves: a training pipeline that runs locally and produces a portable model artifact, and a deployment layer that serves predictions from the cloud.

### Training pipeline

Raw clinical data flows through cleaning and normalization, into a network that learns from it through repeated forward and backward passes, and the trained weights are saved to a JSON file.

```
UCI Heart Disease Data
        ↓
Preprocessing (clean, normalize, split)
        ↓
Neural Network (13 → 8 → 1)
        ↓
Training Loop (backpropagation, 200 epochs)
        ↓
trained_network.json  (portable artifact)
```

### Deployment layer

A client sends patient features as JSON to a public API Gateway endpoint, which routes to an AWS Lambda function that loads the trained weights, runs inference, and returns a structured prediction.

```
Client → API Gateway (POST /predict) → AWS Lambda → JSON Response
```

The trained weights file is bundled into the Lambda deployment package, so the cloud function makes predictions using exactly the same network that was trained locally.

---

## 3. The Network

### Structure

The network is a feedforward architecture with one hidden layer:

| Layer | Neurons | Inputs each | Parameters |
|-------|---------|-------------|------------|
| Hidden | 8 | 13 | 112 |
| Output | 1 | 8 | 9 |
| **Total** | | | **121** |

Each neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a sigmoid activation function that maps any value to a probability between 0 and 1.

### Core math

Four mathematical functions form the foundation, all implemented from scratch:

The **sigmoid** function maps any real number to the range 0 to 1, representing how strongly a neuron activates. The **sigmoid derivative** measures the slope of the sigmoid, which is what makes learning possible during backpropagation. **Mean squared error** measures how wrong the network's predictions are. Its **derivative** points toward the weight adjustments that reduce that error.

### Forward pass

Data moves in one direction — from input to output. Each layer transforms the data and passes its results to the next. The 13 input features enter the hidden layer, each of the 8 hidden neurons produces an activation, and those 8 activations feed the single output neuron, which produces the final prediction.

### Backpropagation

Training works by measuring the error at the output and propagating it backward through the network to calculate how much each individual weight contributed to that error. Each weight is then nudged in the direction that reduces the error, scaled by a learning rate. Over thousands of these adjustments across 200 epochs, the weights settle into values that produce accurate predictions.

---

## 4. The Data

The model trains on the UCI Heart Disease Dataset (Cleveland subset), a real clinical dataset used in published medical research. After removing rows with missing values, 297 patient records remain, each with 13 features:

age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate achieved, exercise-induced angina, ST depression, ST slope, number of major vessels, and thalassemia indicator.

The original target ranges from 0 (no disease) to 4 (severe disease). For this binary classification task, any value above 0 is treated as "abnormal."

### Preprocessing

**Normalization** scales every feature to the 0–1 range using min-max scaling. Without this, a feature ranging from 0 to 300 (like cholesterol) would dominate one ranging from 0 to 1, and the network would struggle to learn. The minimum and maximum of each feature are saved alongside the weights so that new patient data at prediction time is normalized on exactly the same scale.

**Train-test split** holds back 20% of the data for testing. The network never sees this data during training, so its performance on the test set is an honest measure of how well it generalizes rather than how well it memorized.

---

## 5. Results

Training the network for 200 epochs at a learning rate of 0.1 produced the following:

| Metric | Before training | After training |
|--------|-----------------|----------------|
| Accuracy | 43.3% | 91.6% (train) |
| Loss | 0.222 | 0.070 |

On the held-out test set the network achieved roughly 73% accuracy with balanced precision and recall around 69%. Threshold analysis showed that lowering the decision threshold to 0.3 raised recall to 81% — meaning the network caught more genuinely sick patients at the cost of more false alarms.

### Why recall matters most here

In a healthcare context, a **false negative** — telling a sick patient they are healthy — is far more dangerous than a **false positive**. A false positive leads to additional tests and temporary worry. A false negative means a sick patient goes undetected. This is why the evaluation emphasizes recall and includes threshold analysis: a real deployment would likely choose a lower threshold to prioritize catching disease over avoiding false alarms.

---

## 6. Deployment

### AWS Lambda

The trained network is packaged into a self-contained Lambda function. The function contains all the network math inline (it does not import the training code) along with the `trained_network.json` weights file. When a request arrives, the function validates the input, normalizes it using the saved scale, runs the forward pass, and returns a structured prediction.

The network is loaded once per Lambda container and cached in a module-level variable, so repeated requests to a warm container skip the file read — a standard cold-start optimization.

### API Gateway

An HTTP API fronts the Lambda function, exposing a public `POST /predict` route. This makes the model callable by anyone over the open internet with no AWS credentials required.

### Request format

```json
POST /predict
{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
```

### Response format

```json
{
  "prediction": 0.004,
  "label": "NORMAL",
  "risk_level": "VERY LOW RISK",
  "risk_description": "Minimal indicators of heart disease",
  "confidence": 99.2,
  "top_influences": [
    {"feature": "Fasting Blood Sugar", "influence": 0.0637},
    {"feature": "Max Heart Rate", "influence": 0.0216},
    {"feature": "Thal", "influence": 0.0079}
  ],
  "disclaimer": "This tool assists clinical decision-making and does not replace professional medical judgment"
}
```

### Input validation

The handler rejects malformed requests with clear error messages and appropriate HTTP status codes: 400 for bad input (missing features, wrong count, non-numeric values), 500 for internal failures, and 200 for successful predictions.

---

## 7. Feature Influence

Each prediction includes the three features that most influenced it, calculated through a simple sensitivity analysis: each feature is temporarily zeroed out, and the resulting change in the prediction measures that feature's influence. A large change means the feature mattered a lot to this particular prediction. This is a lightweight, model-agnostic version of the feature importance techniques used in production ML systems to make predictions explainable.

---

## 8. Project Structure

```
neural-network-health-detection/
│
├── neural_network.py          Main training pipeline (all phases)
├── lambda_handler.py          Self-contained inference function
├── trained_network.json       Saved weights + normalization params
├── processed.cleveland.data   UCI dataset
└── README.md
```

---

## 9. Known Limitations and Honest Notes

This is a learning project and reflects deliberate tradeoffs.

The network uses a single hidden layer and sigmoid activation throughout. Modern architectures would typically use ReLU activations and possibly deeper structures, which can train faster and reach higher accuracy. Sigmoid was chosen here because its derivative is clean to implement by hand and the math is easier to reason about while learning.

The test-set accuracy (around 73%) is lower than the training accuracy (91.6%), indicating some overfitting. With only 297 records and 121 parameters, the network learns the training set more precisely than it generalizes. Regularization or more data would narrow this gap.

The model is a decision aid, not a diagnostic tool. It is trained on a small historical dataset and should never be used for actual medical decisions.

---

## 10. What This Project Demonstrates

Building a neural network from scratch and deploying it end to end touches the full lifecycle of an applied machine learning system: data cleaning and normalization, model architecture, the training loop and gradient-based optimization, evaluation with metrics chosen for the problem domain, model persistence, and cloud deployment as a scalable inference endpoint.

Every pattern here maps to production ML work. The preprocessing is a data pipeline. The training loop is gradient descent. The evaluation is model validation. The Lambda handler is a real inference service. Building each one by hand means understanding what the libraries and platforms are actually doing — which is the difference between using ML tools and engineering ML systems.
