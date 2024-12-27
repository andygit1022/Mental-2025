########### util.py ##############
import pandas as pd
import params as PARAMS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_data():
    df = pd.read_csv(PARAMS.DATASET_PATH, encoding_errors="ignore")
    columns = ["Label"] + PARAMS.FEATURES
    df = df.astype(PARAMS.FULL_FEATURES)
    df = df[columns]
    max_lengths = df.applymap(lambda x: len(str(x))).max()

    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['Label'])

    # train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'])

    return train_df, val_df

