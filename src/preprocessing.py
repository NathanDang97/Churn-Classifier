import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# helper method that extract unwanted features in the training set
def features_to_drop(train_df, var_threshold=0.1):
    # drop CustomerID which is an irrelevant feature
    cols_to_drop = ['CustomerID']

    # drop low-variance features
    selector = VarianceThreshold(threshold=var_threshold)
    train_df_numeric = train_df.select_dtypes(include=['float64', 'int64'])
    selector.fit(train_df_numeric.drop('Churn', axis=1))
    low_variance_cols = train_df_numeric.drop('Churn', axis=1).columns[~selector.get_support()]
    cols_to_drop.extend(low_variance_cols)

    return cols_to_drop

def data_cleaning(df, cols_to_drop):
    # drop unwanted features
    df.drop(cols_to_drop, axis=1, inplace=True)

    # binary encoding
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['PaperlessBilling', 'MultiDeviceAccess', 'ParentalControl', 'SubtitlesEnabled', 'Gender']:
        df[col] = df[col].map(binary_map)

    # one-hot encoding
    one_hot_cols=['SubscriptionType', 'PaymentMethod', 'ContentType', 'DeviceRegistered', 'GenrePreference']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    return df

def feature_engineering(df):
    # Ratio features
    df['SupportRate'] = df['SupportTicketsPerMonth'] / (df['AccountAge'] + 1)
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['AccountAge'] + 1)

    # Interaction Features
    df['CostEfficiency'] = df['ViewingHoursPerWeek'] / (df['MonthlyCharges'] + 1)
    df['EngagementScore'] = df['ViewingHoursPerWeek'] * df['AccountAge']

    # Combine boolean features into counts 
    # (which may help save some training time as we combine mult features into one)
    df['NumEnabledFeatures'] = (df['MultiDeviceAccess'] + df['ParentalControl'] 
                                + df['SubtitlesEnabled'] + df['PaperlessBilling'])
    
    # Log transform for skewed features
    df['LogTotalCharges'] = np.log1p(df['TotalCharges'])

    # Drop columns related to the engineered features as they now become redundant
    cols_to_drop = ['TotalCharges', 'SupportTicketsPerMonth', 'ViewingHoursPerWeek', 'MultiDeviceAccess', 
                    'ParentalControl', 'SubtitlesEnabled', 'PaperlessBilling']
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def train_val_split(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        stratify=y, 
                                                        random_state=42)

    return X_train, X_val, y_train, y_val

def features_scaling(X_train, X_val, test_df):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_val)
    test_df_scaled = scaler.transform(test_df)
    
    return X_train_scaled, X_test_scaled, test_df_scaled