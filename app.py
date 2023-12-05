from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData, PredictPipline

application = Flask(__name__)

app = application

## Route
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictData", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("/home.html")
    else:
        print(request.form)
        data = CustomData(
           gender=request.form.get("gender"),
           race=request.form.get("race"),
           parental_level_education=request.form.get("parental_level_education"),
           lunch=request.form.get("lunch"),
           test_preparation_course=request.form.get("test_preparation_course"),
           reading_score=request.form.get("reading_score"),
           writing_score=request.form.get("writing_score")
        )
       
       
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipline()
        results = predict_pipeline.predict(pred_df)
        
        print(results)
       
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=3000)