from data_collect import data_collect
from eda import eda
from preprocessing import preprocessing
from modeling import modeling
from evaluation import evaluate

import pickle

if __name__ == "__main__":
    df = data_collect()
    print(df.head())
    df.to_csv("data/raw_data.csv", index=False)
    
    desc, corr = eda(df)
    desc.to_csv("data/describe.csv", index=False)
    corr.savefig("plots/corr.png")
    
    X_train, X_test, y_train, y_test = preprocessing(df)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    print(X_train.shape, X_test.shape)
    
    model = modeling(X_train, y_train)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print(model)
        
    acc = evaluate(model, X_test, y_test)
    print(f"Acurácia: {acc}")