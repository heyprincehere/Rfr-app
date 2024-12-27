import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Prediction App")

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

# User input
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5,1.0)
# Predict
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f"Predicted class: {iris.target_names[prediction[0]]}")