
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ver valores nulos
data = pd.read_csv ('encuesta_bienestar.csv')
Dataframe= pd.DataFrame(data)
print(Dataframe.isnull())

# Eliminar filas con datos faltantes
dataframe_sin_null= Dataframe.dropna()
print(dataframe_sin_null)

# Datos Duplicados
DataframeDupicados= pd.DataFrame(data)
print(DataframeDupicados.duplicated())
print(DataframeDupicados.drop_duplicates())

#Analisis univariado y bivariados:age,hours_sleep, exercise_days, ,happiness_level
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columnas = ['age', 'hours_sleep', 'exercise_days', 'happiness_level']
df = Dataframe[columnas].dropna()

# Estadísticas descriptivas
print(" Estadísticas:")
print(df.describe())

# Gráficos univariados
plt.figure(figsize=(12, 8))
for i, col in enumerate(columnas):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribución de {col}')
plt.tight_layout()
plt.show()

# Grafico Pairplot
columnas = ['age', 'hours_sleep', 'exercise_days', 'happiness_level']
df = df[columnas].dropna()


# Relacion entre edad y dias de ejercicio

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(Dataframe['age'], bins=20, kde=True)  
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.savefig('Edad_ejercicio.png')
plt.show()

# Comparacion entre genero y pais

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.countplot(x='country', hue='gender', data=Dataframe)
plt.title('Cantidad de personas por país y género')
plt.xlabel('País')
plt.ylabel('Cantidad')
plt.legend(title='Género')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Genero_Pais.png')
plt.show()

# -Crear indicadores derivados: ratio screen/sleep, hábitos saludables, etc.
df = pd.read_csv("encuesta_bienestar.csv")
df["screen_sleep_ratio"] = df["screen_time"] / df["hours_sleep"]


print(df["screen_sleep_ratio"])





# Visualizacion de relaciones entre variables

# 1.Correlacion

#Vamos a seleccionar solo las columnas numericas
df_numerico= df.select_dtypes(include=['float64','int64'])

sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de correlaciones')
plt.savefig('Mapa_correlaciones.png')
plt.show()

# 2. Dispersion
sns.scatterplot(x='age',y= 'stress_level',hue='gender',data=Dataframe)
#usaremos hue para colorear los puntos segun sexo
plt.title('Relacion entre edad y nivel de stress')
plt.savefig('relacion_edad_stress.png')
plt.show()

# 3. Barras
sns.barplot(x='gender', y='happiness_level',data= Dataframe)
plt.title('Tasa de nivel de felicidad por sexo')
plt.ylabel('Proporcion de nivel de felicidad')
plt.savefig('Felicidad_por_genero.png')
plt.show()

# Detectar grupos de riesgo (alta pantalla,poco sueño,stress)
grupo_riesgo = df[
    (df['hours_sleep'] < 6) &
    (df['exercise_days'] <= 1) &
    (df['screen_time'] > 6) &
    (df['happiness_level'] <= 2)
]
print(f"Total en riesgo: {len(grupo_riesgo)} personas")
print(grupo_riesgo.head())

sns.pairplot(grupo_riesgo[['hours_sleep', 'exercise_days', 'screen_time', 'happiness_level']])
plt.suptitle('Distribución de indicadores en grupo de riesgo', y=1.02)
plt.savefig('Grupos de riesgo')
plt.show()

# Grupo de riesgo personalizado.
grupo_riesgo = df[
    (df['screen_time'] > 6) &
    (df['hours_sleep'] < 6) &
    (df['stress_level'] >= 4)
]

print(f"Cantidad de personas en riesgo: {len(grupo_riesgo)}")
print(grupo_riesgo.head())

sns.pairplot(grupo_riesgo[['screen_time', 'hours_sleep', 'stress_level', 'happiness_level']])
plt.suptitle('Grupo en riesgo: pantalla alta + poco sueño + alto estrés', y=1.02)
plt.savefig('grupos de riego personalizado')
plt.show()










