import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ucimlrepo import fetch_ucirepo 

import streamlit as st

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBRegressor(n_estimators=500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(X.columns)
# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# # Iniciar a interface Streamlit
st.title('Previsão de Qualidade do Vinho')

st.write("Insira as características do vinho para exibir a qualidade:")

# Slider para as características do vinho (usuário insere os valores)
fixed_acidity = st.slider('Acidez fixa', min_value=0.0, max_value=15.0, value=7.4)
volatile_acidity = st.slider('Acidez volátil', min_value=0.0, max_value=1.5, value=0.7)
citric_acid = st.slider('Ácido cítrico', min_value=0.0, max_value=1.0, value=0.0)
residual_sugar = st.slider('Açúcar residual', min_value=0.0, max_value=15.0, value=1.9)
chlorides = st.slider('Cloretos', min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = st.slider('Dióxido de enxofre livre', min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.slider('Dióxido de enxofre total', min_value=0.0, max_value=200.0, value=34.0)
density = st.slider('Densidade', min_value=0.99, max_value=1.05, value=0.9978)
pH = st.slider('pH', min_value=2.5, max_value=4.0, value=3.51)
sulphates = st.slider('Sulfatos', min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.slider('Álcool', min_value=8.0, max_value=15.0, value=9.4)

input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

predicted_quality = model.predict(input_data)[0]

# Exibir a previsão de qualidade do vinho
st.write(f"A qualidade prevista do vinho é: {predicted_quality:.2f}")