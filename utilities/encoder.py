import pandas as pd


def encode_impl(features, method="float"):
    encoded_features = features.copy()
    if method == "float":
        # Convert categorical variables to float
        for col in encoded_features.select_dtypes(include=['category']):
            encoded_features[col] = encoded_features[col].cat.codes.astype(
                float)
            encoded_features.loc[encoded_features[col]
                                 == -1, col] = float('nan')
    elif method == "dummy":
        # One-hot encoding
        encoded_features = pd.get_dummies(encoded_features, dummy_na=True)
        encoded_features = encoded_features.astype(
            float)  # Convert boolean columns to float
    else:
        raise NotImplementedError("Unsupported encoding method")

    return encoded_features
