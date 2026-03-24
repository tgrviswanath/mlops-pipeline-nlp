import pandas as pd
from sklearn.model_selection import train_test_split

def load_sample_data():
    data = [
        ("I love this product", "positive"),
        ("This is amazing", "positive"),
        ("Best purchase ever", "positive"),
        ("Excellent quality", "positive"),
        ("I hate this", "negative"),
        ("Terrible experience", "negative"),
        ("Waste of money", "negative"),
        ("Very disappointed", "negative"),
        ("It's okay", "neutral"),
        ("Not bad", "neutral"),
    ] * 50
    df = pd.DataFrame(data, columns=["text", "label"])
    return df

def prepare_data(test_size=0.2, random_state=42):
    df = load_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test
