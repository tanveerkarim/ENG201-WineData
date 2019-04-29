
# coding: utf-8

# In[103]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

import seaborn as sns


# In[14]:


wine_q = pd.read_csv("wine-quality_preprocessed.csv")
print("Data dim:",wine_q.shape)
wine_q.head()


# In[78]:


#split data train-test, balance based on logit_score
X= wine_q[wine_q.columns.difference(['logit_score','type','Unnamed: 0'])]
y = wine_q.logit_score
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.20, random_state=42, stratify= wine_q.logit_score)
print("Train Dim:",X_train.shape)
print("Test Dim:",X_test.shape)


# In[79]:



print("Percentage of Good Wines:",np.sum(y_test)/y_test.shape[0]*100,"%")
print("Percentage of Bad Wines:",np.count_nonzero(y_test==0)/y_test.shape[0]*100,"%")


# ## Data analysis

# In[99]:


def correlation_plot(data):
    corr=data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()
    return corr


# In[102]:


corr = correlation_plot(X)


# In[105]:


# plot the heatmap
tmp=sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# ## Remove Correlated Variables?

# ## Logisitc Regression Approach

# In[86]:


#Logistic Regression Model
model_lr = LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr')
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Acc:",acc*100,"%")

print("Predictor Coefficients:\n",model_lr.coef_)

print("\nTN","FP\n")
print("FN","TP")
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))


# In[84]:


#Logistic Regression with Cross Validation
model_lr_cv = LogisticRegressionCV(cv=5, random_state=42, solver='liblinear',multi_class='ovr')
model_lr_cv.fit(X_train, y_train)

y_pred = model_lr_cv.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Logistic Regression w. CV Acc:",acc*100,"%")
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))


# ## SVM Approach

# In[89]:


#SVM
model_svm = svm.LinearSVC(multi_class='ovr', penalty='l2', random_state=42,max_iter=1000)
model_svm.fit(X_train,y_train)

y_pred = model_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("SVM Acc:",acc*100,"%")
print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred, labels=None, sample_weight=None))


# ## Note classification has to be higher than 83 as 83% if the data belong to 1 class... note for text processing
