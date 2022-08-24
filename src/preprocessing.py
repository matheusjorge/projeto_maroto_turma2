from sklearn.model_selection import train_test_split

def split(df, test_size=0.1):
    X = df.drop(columns=["target"])
    y = df["target"]
    
    return train_test_split(X, y, test_size=test_size)

def preprocessing(df):
    X_train, X_test, y_train, y_test = split(df)
    
    return X_train, X_test, y_train, y_test