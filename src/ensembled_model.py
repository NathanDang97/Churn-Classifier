from itertools import product
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def ensemble_model(y_val, y_pred_probs_nn, y_pred_probs_rf, y_pred_probs_xgb):
    # We have predicted probabilities from the 3 models:
    # y_pred_probs_nn  : from neural network
    # y_pred_probs_rf  : from random forest
    # y_pred_probs_xgb : from xgboost
    # y_val            : labels for evaluation

    # Normalize weights so they sum to 1
    weight_options = np.arange(0.0, 1.1, 0.1)

    best_auc = 0
    best_weights = (0, 0, 0)

    for w1, w2 in product(weight_options, repeat=2):
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue  # invalid weight combination

        # Weighted average of probabilities
        probs_ensemble = w1 * y_pred_probs_nn + w2 * y_pred_probs_rf + w3 * y_pred_probs_xgb

        # Evaluate
        auc = roc_auc_score(y_val, probs_ensemble)

        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2, w3)

    best_probs_ensembled = best_weights[0] * y_pred_probs_nn + best_weights[1] * y_pred_probs_rf + best_weights[2] * y_pred_probs_xgb

    return (best_weights, best_probs_ensembled)

def evaluate_ensembled_model(model, y_val):
    best_weights, best_probs_ensembled = model

    # Predict labels and calculate ROC AUC
    y_pred_ensemble = (best_probs_ensembled > 0.5).astype(int)
    auc = roc_auc_score(y_val, best_probs_ensembled)

    # evaluate
    print("\n--- Ensembled Model Evaluation ---")
    print("Best Weights (NN, RF, XGB):", best_weights)
    print(confusion_matrix(y_val, y_pred_ensemble))
    print(classification_report(y_val, y_pred_ensemble))
    print(f"ROC AUC: {auc:.4f}")