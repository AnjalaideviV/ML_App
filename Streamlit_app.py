
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Streamlit')


# Title for the Streamlit app
st.title('Titanic Survival Prediction')

submit= st.button('Click here')

if submit:
     st.dataframe(train)



# Data Exploration:
st.header('Data Exploration:')

# Load the dataset
train = pd.read_csv('Titanic_train.csv')
st.write('Data Info')
st.dataframe(train)

# Show data statistics
st.write('Summary')
table=train.describe().T
st.dataframe(table)

# Histograms
st.subheader('Histogram of Features')
fig, ax = plt.subplots(figsize=(15, 10))
train.hist(ax=ax, bins=30, edgecolor='black')
st.pyplot(fig)

# Boxplot
st.subheader('Boxplot')
fig, ax = plt.subplots(figsize=(18,8))
train.boxplot(ax=ax)
st.pyplot(fig)

# Data Preprocessing:
st.header('Data Preprocessing:')
info=train.info();
st.table(info)

st.write('Missing values in each column:')
st.write(train.isnull().sum())

mean = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean)

# Model Building:
st.header('3. Model Building')

df = train[['Survived', 'Pclass', 'Sex', 'Age']]
df = pd.get_dummies(df, columns=['Pclass', 'Sex']).astype(int)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

y = df['Survived']
x = df.drop('Survived', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)

# Model Evaluation:
st.header('4. Model Evaluation')

st.write('Model Accuracy:', model.score(x_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report

y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
st.write('Confusion Matrix:')
st.write(cm)

cr = classification_report(y_test, y_predicted)
st.write('Classification Report:')
st.write(cr)


