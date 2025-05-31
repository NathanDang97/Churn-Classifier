import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# drop out probability
DROP_OUT = 0.3

def build_nn(input_dim, output_dim=1):
    # define the model architecture
    model = Sequential()

    # input layer + hidden 1
    model.add(Dense(256, input_shape=input_dim))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(DROP_OUT))

    # hidden 2
    model.add(Dense(128))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(DROP_OUT))

    # hidden 3
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(DROP_OUT))

    # output layer
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

def train_nn(model, X_train, y_train, epochs=50, batch_size=64):
    # compute class weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    # define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

def evaluate_nn(model, X_val, y_val):
    y_pred_proba = model.predict(X_val).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    auc = roc_auc_score(y_val, y_pred_proba)

    print("\n--- Neural Network Evaluation ---")
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"ROC AUC: {auc:.4f}")

    return y_pred_proba