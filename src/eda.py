import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def describe(df):
    return df.describe()

def correlation(df):
    fig, ax = plt.subplots(figsize=(16,9))
    sns.heatmap(df.drop(columns=["target"]).corr(), cmap="Blues", annot=True, ax=ax)
    
    return fig

def eda(df):
    desc = describe(df)
    corr = correlation(df)
    
    return (
        desc, 
        corr
    )