from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def xgb_classifier(X_train, y_train, lr=0.01):
    # split the full training set to a training set and validation set for training XGB
    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # compute the scale between positive and negative labels
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # Initialize XGB
    xgb_clf = XGBClassifier(
        n_estimators=1000,
        early_stopping_rounds=5,
        max_depth=6,
        learning_rate=lr,
        scale_pos_weight=scale_pos_weight,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )

    # Train the model
    xgb_clf.fit(X_train_xgb, y_train_xgb,
                eval_set=[(X_val_xgb, y_val_xgb)],
                verbose=False
            )
    
    return xgb_clf

def rf_classifier(X_train, y_train, max_depth=10):
    # compute class weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Initialize Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=max_depth,
        random_state=42,
        class_weight=class_weights,
        n_jobs=-1
    )

    # Train the model
    rf_clf.fit(X_train, y_train)

    return rf_clf

def evaluate_clf(model, X_val, y_val, label):
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- {label} Evaluation ---")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    if y_pred_proba is not None:
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"ROC AUC: {auc:.4f}")

    return y_pred_proba