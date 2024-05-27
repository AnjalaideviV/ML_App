
import matplot.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

# Title
st.title("Titanic Survival Prediction")

# Load Data
@st.cache
def load_data():
    return pd.read_csv('Titanic_train.csv')

train = load_data()

# Data Exploration
st.header("Data Exploration")
st.write(train.head())
st.write(train.tail())
st.write(train.describe().T)

fig, ax = plt.subplots(figsize=(15, 10))
train.hist(bins=30, edgecolor='black', ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
train.boxplot(ax=ax)
st.pyplot(fig)

# Data Preprocessing
st.header("Data Preprocessing")
train.info()
st.write(train.isnull().sum())

mean = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean)
st.write("Mean Age:", mean)

st.write("Number of Duplicates:", train.duplicated().sum())

df = train[['Survived', 'Pclass', 'Sex', 'Age']]
df = pd.get_dummies(df, columns=['Pclass', 'Sex']).astype(int)

# Model Building
st.header("Model Building")
y = df['Survived']
x = df.drop('Survived', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
st.write(f"Model Accuracy: {accuracy}")

# Model Evaluation
st.header("Model Evaluation")
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
st.write("Confusion Matrix:", cm)

predictions = cross_val_predict(model, x_train, y_train, cv=3)
cm_train = confusion_matrix(y_train, predictions)
st.write("Cross-Validation Confusion Matrix:", cm_train)

cl_report = classification_report(y_test, y_predicted)
st.write("Classification Report:", cl_report)

