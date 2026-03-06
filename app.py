from flask import Flask,render_template,request
import pickle
import numpy as np
model=pickle.load(open("model.pkl","rb"))
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict():
    itching=int(request.form["itching"])
    skin_rash=int(request.form["skin_rash"])
    nodal_skin_eruptions=int(request.form["nodal_skin_eruptions"])
    continuous_sneezing=int(request.form["continuous_sneezing"])
    features=np.array([[itching,skin_rash,nodal_skin_eruptions]])
    prediction=model.predict(features)
if __name__=="__main__":
    app.run(debug=True)
