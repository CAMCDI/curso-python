import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# crear datos ficticios
n=100  # numero de nuestras

#general valores para la cantidad de fertilizante (x) eun un rango de 0 a 100 kg
np.random.sedd(42) #reproducibilidad
fertilizante = np.random.uniform(0,100,n)

#definir una relacion lineal con ruido aleatorio para la cantidad de papas
#suponemos que por cada kg de fertilizante obtendremos 0.8 toneladas
# y agregamos un oco de ruido normal para simular variedad normaal 
papas = 0.8 * fertilizante + np.random.normal(0,5,n) + 8 # intersecto en 8 

df = pd.DataFrame ({
    'Fertilizante (kg)': fertilizante,
    'Papas (toneldas)': papas
})

X = df[['Fertilizante' (kg)]].values
Y = df['Papas (toneladas)'].values

#separar los datos de netrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#definir modelo y entrenarlo
regressor = LinealRegression()
regressor.fit(X_train,y_train)

#hacer la prediccion 
y_pred = regressor.predict(X_test)






