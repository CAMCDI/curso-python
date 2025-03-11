import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from pandasgui import show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# crear datos ficticios
n=100  # numero de nuestras

#general valores para la cantidad de fertilizante (x) eun un rango de 0 a 100 kg
np.random.sedd(42) #reproducibilidad
fertilizante = np.random.uniform(0,100,n)

#definir una relacion lineal con ruido aleatorio para la cantidad de papas
#suponemos que por cada kg de fertilizante obtendremos 0.8 toneladas
# y agregamos un poco de ruido normal para simular variedad normaal 
papas = 0.8 * fertilizante + np.random.normal(0,5,n) + 8 # intersecto en 8 

df = pd.DataFrame ({
    'Fertilizante (kg)': fertilizante,
    'Papas (toneldas)': papas
})

X = df[['Fertilizante' (kg)]].values
Y = df['Papas (toneladas)'].values

#separar los datos de tratamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#definir modelo y entrenarlo
regressor = LinealRegression()
regressor.fit(X_train,y_train)

#hacer la prediccion 
y_pred = regressor.predict(X_test)

# calcular b0 y b1
b1 = regressor.coef_[0]
b0 = regressor.intercept_

print(f"""
    b1: {b1}
    b0) {b0}
""")

# calcular mse. rmse, r^2
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"""
    mse: {mse}
    rmse: {rmse}
    r2: {r2}
""")


plt.figure(figsize=(10,6))

# scatter plot del set de entrenamiento 

plt.scatter(X_train,y_train, color="blue", label="datos de entrenamiento")

# scatter plot para el set de prueba
plt.scatter(X_test,y_test, color="green", label="Datos de prueba")

plt.plot(X_train,regressor.predict(X_train), color="read", lable="linea de regresion")

plt.title("regresion lineal: fertilizante vs produccion de papas")

plt.xlabel("fertilizante (kg)")
plt.ylabel("papas (toneladas)")
plt.legend()

plt.show()










