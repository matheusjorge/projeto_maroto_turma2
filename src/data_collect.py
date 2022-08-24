import pandas as pd
from sklearn.datasets import load_wine

def data_collect():
    data = load_wine()
    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    
    return df

if __name__ == "__main__":
    df = data_collect()