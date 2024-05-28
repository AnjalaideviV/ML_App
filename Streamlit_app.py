import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Streamlit')

# Title for the Streamlit app
st.title('Titanic Survival Prediction')

# Load the dataset
train = pd.read_csv('Titanic_train.csv')
st.write('For Data Information')


submit= st.checkbox('Click here')
if submit:
     st.dataframe(train)

# Data Exploration:
st.header('1.Data Exploration:')


# Show data statistics
table=train.describe().T
st.write('For Summary')
submit= st.button('Click here')
if submit:
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


st.subheader('Missing Values')
missing_data = {
    'Feature': ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare'],
    'Missing Values': [0, 0, 0, 0, 0, 177, 0, 0, 0, 0]
}
missing_df = pd.DataFrame(missing_data)
selected_feature = st.selectbox('Select a feature to view missing values', missing_df['Feature'])
missing_count = missing_df[missing_df['Feature'] == selected_feature]['Missing Values'].values[0]
st.write(f"Missing values in {selected_feature}: {missing_count}")
mean = train['Age'].mean()
train['Age'] = train['Age'].fillna(mean)

# Model Building:
st.header('2. Model Building')
df = train[['Survived', 'Pclass', 'Sex', 'Age']]
df = pd.get_dummies(df, columns=['Pclass', 'Sex']).astype(int)
st.dataframe(df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
y = df['Survived']
x = df.drop('Survived', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LogisticRegression()
st.write('Model:',model.fit(x_train, y_train))

# Model Evaluation:
st.header('4. Model Evaluation')
st.write('Model Accuracy:', model.score(x_test, y_test))
from sklearn.metrics import confusion_matrix, classification_report
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)
st.write('Confusion Matrix:')
st.write(cm)



report_dict = classification_report(y_test, y_predicted, output_dict=True)
report_str = classification_report(y_test, y_predicted)
report_df = pd.DataFrame(report_dict).transpose()
option = st.selectbox(
    'How would you like to view the classification report?',
    ('Table', 'JSON', 'Plain Text')
)
st.write('Classification Report:')

if option == 'Table':
    st.table(report_df)
elif option == 'JSON':
    st.json(report_dict)
else:
    st.text(report_str)

y_score=model.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

st.subheader('ROC Curve')
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.legend(loc="lower right")

# Display the plot in Streamlit
st.pyplot(fig)
