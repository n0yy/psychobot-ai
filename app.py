from flask import Flask, request, jsonify
from src.model import load, predict
import pandas as pd
from src.llm import make_prompt, get_summary

app = Flask(__name__)
model = load("./model/lgbm.pkl")

@app.route("/predict", methods=["post"])
def predict_api():
    try:
        req = request.json
        data = {
            "Age": [req["Age"]],
            "Gender": [req["Gender"]],
            "Platform": [req["Platform"]],
            "Daily_Usage_Time (minutes)": [req["Daily_Usage_Time (minutes)"]],
            "Posts_Per_Day": [req["Posts_Per_Day"]],
            "Likes_Received_Per_Day": [req["Likes_Received_Per_Day"]],
            "Comments_Received_Per_Day": [req["Comments_Received_Per_Day"]],
            "Messages_Sent_Per_Day": [req["Messages_Sent_Per_Day"]]
        }
        
        df = pd.DataFrame(data)
        prediction, chart_data = predict(df, model)
        # Generate Repersentatio of Predictio
        prompt = make_prompt(chart_data)
        md = get_summary(prompt)
        
        res = {
            "predicition": prediction.tolist(),
            "chart_data": chart_data.to_dict(orient="records"),
            "repersentation": md
        }
        
        

        return jsonify(res), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400
        
if __name__ == "__main__":
    app.run()
