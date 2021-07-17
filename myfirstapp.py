import streamlit as st
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('./in/income.csv')
# st.write(df.columns)
st.image('./pig.jpg')

siteHeader = st.beta_container()
with siteHeader:
    #Titulo
    st.title('Modelo de Evaluación de ingresos')
    st.markdown(""" El objetivo de este proyecto es proveer una herramienta que nos permita **predecir** si una persona ganará más o menos de $50k anuales""")
    
dataExploration = st.beta_container()
with dataExploration:
    #subtitulo del contenido
    st.subheader('Dataset:Ingresos')
    st.text("Utilizaremosla trnformacion del sigueinte dataset:")
    st.dataframe(df.head())

dataViz = st.beta_container()
with dataViz:
    st.subheader("Exploración de la data:")
    st.text("Distribución de los datos por sexo")
    st.area_chart(df.sex.value_counts())
    st.text("Distribución de los datos respecto a la edad")
    st.bar_chart(df.age.value_counts())

newFeatures = st.beta_container()
with newFeatures:
    st.subheader("Selección de variables")
    st.markdown(""" El modelo trabaja con las vars **race, sex, workclass, education**""")
    st.text("Quieres considerar otra variables?")
    
optional_cols = ['education-num','marital-status','occupation','relationship']
# Widget para seleccionar
options = st.multiselect('Variables que se añadirán al modelo:',
     optional_cols)

principal_columns = ['race','sex','workclass','education']
drop_columns = ['income','fnlwgt','capital-gain','capital-loss','native-country','income_bi']

if len(options)!=0:
    principal_columns = principal_columns + options
    drop_columns = drop_columns + [i for i in optional_cols if i not in options]
else:    
    drop_columns = drop_columns + optional_cols

# condicional: sirve para hacer validación de columnas
# aparte de estar desplegando esta lista, la estamos guardando en una variable options que nos ayuda a ver cuáles listas vamos a considerar y cuáles no

modelTraining = st.beta_container()
with modelTraining:
    st.subheader('Entrenamiento del modelo')
    st.text('En esta sección puedes selccionar los parámetros del modelo')

#Definimos nuestras variables
Y = df['income_bi']
df = df.drop(drop_columns, axis=1)
X = pd.get_dummies(df, columns = principal_columns)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size =.2, random_state=15)

#es una barra que nos permite preguntar al usuario la info que le agregamos, nospermite seleccionar parámetros
max_depth = st.slider('¿Cuál debería ser el valor de max_depth para el modelo?', min_value=1, max_value = 10, value =7, step =1)

# Modelo 
t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
model = t.fit(X_train, y_train)

#perfirmance del modelo

score_train = model.score(X_train, y_train)
score_test = model.score(X_test,  y_test)

Performance = st.beta_container()
with Performance:
    st.subheader('Performance del modelo:')
    # nos divide la pantalla enel número de columnas que le pasamos
    col1, col2 = st.beta_columns(2)
    with col1:
        st.text('Score Train')
        st.text(round(score_train*100,2))
        
# nuestro modelo va cambiando, porque va entrenándose con lo que mete el usuario