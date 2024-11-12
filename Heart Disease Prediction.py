from google.colab import drive
mount = '/content/drive'
drive.mount(mount)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.optimize as opt 
import statsmodels.api as sm 
from sklearn import preprocessing  
file_path = '/content/drive/MyDrive/dataset.csv'
disease_df = pd.read_csv(file_path)
disease_df.dropna(axis = 0, inplace = True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())
X = np.asarray(disease_df[['age', 'male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])
scaler = preprocessing.StandardScaler()
scaler.fit(X) 
X = scaler.transform(X) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
fig, ax = plt.subplots(figsize=(8, 5))  
sns.countplot(x='TenYearCHD', data=disease_df, ax=ax)
ax.set_title('Count of TenYearCHD')  
ax.set_xlabel('Data')  
ax.set_ylabel('Count')  
plt.show()
p = disease_df['TenYearCHD'].plot()
plt.show(p)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#Defining the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('lr', LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])

plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")

plt.show()

print (classification_report(y_test, y_pred))

#Calculating accuracy of the model..
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', 
      accuracy_score(y_test, y_pred))
