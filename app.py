import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from PIL import Image
#app=Flask(name)
#Swagger(app)
import os
import joblib
rf_regressor = joblib.load("rf_model_red.joblib")
d_red = pd.read_csv('winequality-red.csv', delimiter=';',header=0)
from sklearn.impute import SimpleImputer
# imputer to replace any Nan or Null values
imputer = SimpleImputer(strategy='mean')
col=d_red.columns #storing the columns of the dataframe as imputer removes it
d_red=pd.DataFrame(imputer.fit_transform(d_red))
d_red.columns=col
from sklearn.preprocessing import MinMaxScaler
# Normalizing the data
scaler = MinMaxScaler()
d_red.iloc[:,:-1] = scaler.fit_transform(d_red.iloc[:,:-1])
#@app.route('/')
def welcome():
    return "Welcome All"
#@app.route('/predict',methods=["Get"])
def predict_note_authentication(fa,va,ca,rs,chlo,fsd,tsd,dens,ph,sulp,alc):
    scaled_arr=scaler.transform([[fa,va,ca,rs,chlo,fsd,tsd,dens,ph,sulp,alc]])
    prediction=rf_regressor.predict(scaled_arr)
    print(prediction)
    return prediction

def main():
    st.title("210110014_Akshat Taparia")
    html_temp = """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:white;text-align:center;">Red wine random forest regressor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    fa = st.select_slider("fixed acidity",np.arange(0, 20, 0.1))
    va = st.select_slider("volatile acidity",np.arange(0, 2, 0.1))
    ca = st.select_slider("citric acid",np.arange(0, 1, 0.1))
    rs = st.select_slider("residual sugar",np.arange(0, 20, 0.1))
    chlo=st.select_slider("chlorides",np.arange(0, 1, 0.1))
    fsd=st.select_slider("free sulphur dioxide",np.arange(0, 60, 1))
    tsd=st.select_slider("total sulphur dioxide",np.arange(0, 200, 1))
    dens=st.select_slider("density",np.arange(0.99, 1.1, 0.0001))
    ph=st.select_slider("pH",np.arange(2.5, 4.5, 0.1))
    sulp=st.select_slider("sulphates",np.arange(0, 1.5 , 0.1 ))
    alc=st.select_slider("alcohol",np.arange(7.5, 15, 0.1))

    if st.button("Predict"):
        result=predict_note_authentication(fa,va,ca,rs,chlo,fsd,tsd,dens,ph,sulp,alc)
    st.success('The output is {}'.format(result[0]))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()