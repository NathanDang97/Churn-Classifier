from data_loader import load_data
from preprocessing import features_to_drop, data_cleaning, feature_engineering, features_scaling, train_val_split
from classifiers import xgb_classifier, rf_classifier, evaluate_clf
from neural_net import build_nn, train_nn, evaluate_nn
from ensembled_model import ensemble_model, evaluate_ensembled_model
import pandas as pd

# work pipeline
def main():
    # 1. load the datasets
    train_df, test_df = load_data()
    test_customer_IDs = test_df[['CustomerID']] # extract customer IDs in the test set for submission file

    # 2. data preprocessing
    # 2.1 data cleaning
    var_threshold = 0.1
    cols_to_drop = features_to_drop(train_df, var_threshold)
    train_df = data_cleaning(train_df, cols_to_drop) # will be used for training and validation
    test_df = data_cleaning(test_df, cols_to_drop) # will be used for creating the submission file (so this is like a hidden test set)

    # 2.2 feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # 2.3 train-val split the full training set
    X_train, X_val, y_train, y_val = train_val_split(train_df)

    # 2.4 feature scaling
    X_train_scaled, X_val_scaled, test_df_scaled = features_scaling(X_train, X_val, test_df)

    # 3. Define, Train, and Evaluate the Classifiers
    xgb_clf = xgb_classifier(X_train, y_train)
    rf_clf = rf_classifier(X_train, y_train)
    y_pred_probs_xgb = evaluate_clf(xgb_clf, X_val, y_val, "XGB Classifier")
    y_pred_probs_rf = evaluate_clf(rf_clf, X_val, y_val, "Random Forest Classifier")

    # 4. Define, Train, and Evaluate the Neural Net Model
    input_dim = (X_train_scaled.shape[1],)
    nn_clf = build_nn(input_dim)
    nn_clf, history = train_nn(nn_clf, X_train_scaled, y_train)
    y_pred_probs_nn = evaluate_nn(nn_clf, X_val_scaled, y_val)

    # 5. Define, Train, and Evaluate the Ensembled Model
    ensembled_clf = ensemble_model(y_val, y_pred_probs_nn, y_pred_probs_rf, y_pred_probs_xgb)
    evaluate_ensembled_model(ensembled_clf, y_val)

    # 6. Create prediction for the hidden test set
    predicted_probs_nn = nn_clf.predict(test_df_scaled).flatten()
    predicted_probs_rf = rf_clf.predict_proba(test_df)[:, 1]
    predicted_probs_xgb = xgb_clf.predict_proba(test_df)[:, 1]
    w1, w2, w3 = ensembled_clf[0] # extract the best weights
    predicted_probability = predicted_probs_nn * w1 + predicted_probs_rf * w2 + predicted_probs_xgb * w3
    prediction_df = pd.DataFrame({'CustomerID': test_customer_IDs.values[:, 0],
                                'predicted_probability': predicted_probability})
    prediction_df.to_csv("prediction_submission.csv", index=False)

    # 7. Test if the submission file is valid
    submission = pd.read_csv("prediction_submission.csv")
    assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_submission.'
    assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
    assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'
    assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'
    assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'

if __name__ == "__main__":
    main()