import pandas as pd
import params as PARAMS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_data():
    df = pd.read_csv(PARAMS.DATASET_PATH)
    columns = ["Type"] + PARAMS.FEATURES
    df[PARAMS.FEATURES] = df[PARAMS.FEATURES].astype(str)
    df = df[columns]

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['Type'])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, val_df
