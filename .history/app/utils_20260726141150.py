import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODEL_FEATURES = [
    "absences",
    "failures",
    "goout",
    "age",
    "Mjob",
    "health",
    "Medu",
    "freetime",
    "Fedu",
    "Fjob",
    "Walc",
    "famrel",
    "studytime",
    "reason",
    "guardian",
    "Dalc",
    "schoolsup",
    "romantic",
    "traveltime",
    "activities",
    "paid",
    "sex",
    "famsize",
    "address",
    "famsup",
    "nursery",
    "higher",
    "internet"
]


def preprocess_data(df):
    """
    Preprocess uploaded student dataset to match
    the format used during model training.
    """

    df = df.copy()

    # Remove columns that were NOT used during training
    columns_to_remove = ["Pass", "G1", "G2", "G3", "school", "Pstatus"]

    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Encode categorical columns
    for column in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    # Keep only the required features
    df = df[MODEL_FEATURES]

    return df