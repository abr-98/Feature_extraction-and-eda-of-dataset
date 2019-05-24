#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=1000)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
road_lookup=dict(road.Zone.unique()))
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
plt.title('training class')
plt.plot(X_train,y_train)
plt.show()


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
'''road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
plt.title('training class')
plt.scatter(X_train,y_train,marker='o')
plt.show()


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,100)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Intersection count','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[111]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
clf=DecisionTreeClassifier(max_depth=3).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[108]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)
clf=DecisionTreeClassifier(max_depth=7).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[493]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Honk_duration','RMS','Intersection count','Timelevel']]
y=road['Zone']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25)
clf=DecisionTreeClassifier(max_depth=5,min_samples_split=2).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[507]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=25,test_size=0.25)
knn=KNeighborsClassifier(n_neighbors=9,p=1,weights='distance')
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances


road=pd.read_csv('dataset.csv')
X=road[['Route','Timelevel','Segment_length','WiFi count','RMS','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=15,test_size=0.25)
clf=DecisionTreeClassifier(max_depth=5,min_samples_split=2,max_leaf_nodes=4000).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))
print(clf.feature_importances_)
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()


# In[490]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=32,p=1)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[197]:


road.head()


# In[509]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Zone']
k_range=np.arange(1,100)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=25)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k,p=1,metric='minkowski')
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']
k_range=np.arange(1,100)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k,p=2,metric='minkowski',weights='uniform')
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[ ]:




