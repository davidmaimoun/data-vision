import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

LINEAR = 'Linear Regression'
LOGISTIC = 'Logistic Regression'
DECISION_TREE = 'Decision Trees'
RANDOM_FOREST = 'Random Forest'
KNN = 'K Nearest Neighbor (KNN)'
SVM = 'Support Vector Machine SVM'
PROPHET = 'Prophet - Automatic Forecasting Procedure'

def createDataExploration(df_ml, model, target):
   sns.set_style("whitegrid")
   
   if model == LINEAR:
      st.markdown(f"<h5 class='title_section'>Distribution of {target}</h5>", unsafe_allow_html=True)
      fig2 = plt.figure(figsize=(10, 4))
      sns.distplot(df_ml[target],
               hist_kws=dict(edgecolor="black", linewidth=1))
      st.pyplot(fig2, use_container_width=True)

   st.markdown(f"<h5 class='title_section'>Data Values Correlation</h5>", unsafe_allow_html=True)
   corr = df_ml.corr()

   fig3 = plt.figure(figsize=(10, 4))
   sns.heatmap(corr, annot = True)
   st.pyplot(fig3, use_container_width=True)

def trainAndTestSplit(df_ml, train_test_values, target, test_size):
   train_idx, test_idx = train_test_split(df_ml.index, test_size=test_size)
   df_ml['split'] = 'train'
   df_ml.loc[test_idx, 'split'] = 'test'

   X_train = df_ml.loc[train_idx, train_test_values]
   X_test = df_ml.loc[test_idx, train_test_values]
   y_train = df_ml.loc[train_idx, target]
   y_test = df_ml.loc[test_idx, target]
   return X_train, X_test, y_train, y_test

def returnMLModel(model , params=None):
   if model == LINEAR:
      from sklearn.linear_model import LinearRegression
      return LinearRegression()
   
   elif model == LOGISTIC:
      from sklearn.linear_model import LogisticRegression
      return LogisticRegression()
   
   elif model == KNN:
      from sklearn.neighbors import KNeighborsClassifier            
      return KNeighborsClassifier(n_neighbors = params['n_neighbors'])
   
   elif model == DECISION_TREE:
      from sklearn.tree import DecisionTreeClassifier
      return DecisionTreeClassifier()

   elif model == RANDOM_FOREST:
      from sklearn.ensemble import RandomForestClassifier
      return RandomForestClassifier(n_estimators=params['n_estimators'])

   elif model == SVM:
      from sklearn.svm import SVC
      return SVC() 
   
def createConfusionMatrix(y_test, prediction):
   return confusion_matrix(y_test, prediction)

def getPairPlot(df_ml, target):
   sns.set_style("whitegrid")
   
   fig = sns.pairplot(df_ml, hue=target,palette='bwr')
   st.pyplot(fig)
