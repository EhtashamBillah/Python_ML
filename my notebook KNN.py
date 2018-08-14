
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Classified Data.csv', index_col = 0) #index_col = 0 means take the 1st column of dataset as index for rows
df.head()


# # centering and scalling

# In[3]:


from sklearn.preprocessing import StandardScaler


# In[4]:


scaler = StandardScaler() # Centering and scaling happen independently on each feature


# In[5]:


scaler.fit(df.drop('TARGET CLASS', axis = 1))


# In[6]:


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))


# In[7]:


df.head(2)


# In[8]:


print(scaled_features)


# In[9]:


df_feat = pd.DataFrame(data = scaled_features, columns=df.columns[:-1])


# In[10]:


df_feat.head(2)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


test_size = 0.30
seed = 101
x = df_feat
y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=seed)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier


# In[14]:


knn = KNeighborsClassifier(n_neighbors=6)


# In[15]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[16]:


kfold = KFold(n_splits= 10, random_state=seed)
results = cross_val_score(knn,x_train,y_train,cv=kfold,scoring = 'roc_auc')


# In[17]:


print("AUC =>" , 'Mean:',results.mean().round(3), '&', 'SD:',results.std().round(3))


# In[18]:


knn.fit(x_train,y_train)


# In[20]:


y_hat = knn.predict(x_test)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix


# In[23]:


print(classification_report(y_test,y_hat))


# In[25]:


print(confusion_matrix(y_test,y_hat))


# In[28]:


# elbo method to find the optimum k

error_rate =[]

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    


# In[74]:


plt.figure(figsize=(10,6))
plt.plot( range(1,40),error_rate,color='b',linestyle='dashed',marker = 'o',markerfacecolor='r')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[75]:


# setting k=17 according to the plot
knn = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(knn,x_train,y_train,cv=kfold,scoring = 'roc_auc')
knn.fit(x_train,y_train)
y_hat = knn.predict(x_test)


# In[76]:


print("AUC =>" , 'Mean:',results.mean().round(3), '&', 'SD:',results.std().round(3))
print(classification_report(y_test,y_hat))
print(confusion_matrix(y_test,y_hat))

