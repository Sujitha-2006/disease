import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

data = {
    "itching":[1,1,0,0],
    "skin_rash":[1,1,0,0],
    "nodal_skin_eruptions":[1,0,0,0],
    "continuous_sneezing":[0,1,0,0],
    "disease":[
        "Fungal infection",
        "Allergy",
        "Healthy",
        "Healthy"
    ]
}
df = pd.DataFrame(data)

X = df[["itching","skin_rash","nodal_skin_eruptions","continuous_sneezing"]]
y = df["disease"]

model = DecisionTreeClassifier()
model.fit(X,y)

pickle.dump(model,open("model.pkl","wb"))

print("Model trained successfully")