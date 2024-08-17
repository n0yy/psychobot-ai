import pandas as pd
import pickle

# Load Model
def load(path: str):
    with open(path, "rb") as file:
        return pickle.load(file)

def predict(df: pd.DataFrame, model):
        
    pred = model.predict(df)
    proba = model.predict_proba(df)[0]
    classes = model.classes_
    
    chart_data = pd.DataFrame({
        "Emotional": classes,
        "Probability": proba * 100
    }).sort_values("Probability", ascending=False).reset_index(drop=True)
    
    return pred, chart_data