#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("A")


# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier


list=pd.read_csv('dataset.csv')

X=list.drop(['Class','Time'],axis=1).values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)
print("c")
X_train,X_test,y_train,y_test=train_test_split(X_d_2,y_d_2,test_size=0.1,random_state=21)
print("d")
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)
print("e")
knn.fit(X_train,y_train)
print("f")

y_pred=knn.predict(X_test)
print("g")
print(y_pred)
print(knn.score(X_test,y_test))


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier


list=pd.read_csv('dataset.csv')

X=list.drop(['Class','Time'],axis=1).values
X_d=pd.DataFrame(X)
y=list['Class','Timelevel'].values
y_d=pd.DataFrame(y)
print("a")
X_d_2=pd.get_dummies(X_d)
print("b")
y_d_2=pd.get_dummies(y_d)
print("c")
X_train,X_test,y_train,y_test=train_test_split(X_d_2,y_d_2,test_size=0.1,random_state=21)
print("d")
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)
print("e")
clf.fit(X_train,y_train)
print("f")
print(clf.score(X_test,y_test))


# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,y_d,test_size=0.2,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[18]:


import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(y)
X_train,X_test,y_train,y_test=train_test_split(X_d,y_d,test_size=0.2,random_state=21)
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


# In[37]:


import pandas as pd
import gmplot

df=pd.read_csv('6mar.csv')
print("c")
list_long=df['Start_Lng'].tolist()
print("d")
list_lat=df['Start_Lat'].tolist()
print("e")
gmap4 = gmplot.GoogleMapPlotter(23.54599549,87.29266503,14) 
gmap4.heatmap(list_long,list_lat ) 

print("a")
gmap4.draw( "home\ml\map1.html" ) 
print("b")


# In[39]:


import pandas as pd
import gmplot

df=pd.read_csv('6mar.csv')
print("c")
list_long=df['Start_Lng'].tolist()
print("d")
list_lat=df['Start_Lat'].tolist()
print("e")
gmap4 = gmplot.GoogleMapPlotter(23.54599549,87.29266503,14) 
gmap4.heatmap(list_long,list_lat ) 

print("a")
gmap4.draw( "map1.html" ) 
print("b")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
X=pd.read_csv('dataset.csv')
plt.figure()
sns.countplot(x='Class',hue='WiFi density',data=X,palette='RdBu')

plt.xticks([0,1,2,3],['slow','normal','fast','very fast'])
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
lasso=Lasso(alpha=0.1)
lasso_coef=lasso.fit(X_d,Y_d_2).coef_
plt.plot(range(len(x)),lasso_coef)
plt.xticks(range(len(x)),x,rotation=60)
plt.ylabel('coefficients')
plt.show()


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
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


# ####

# In[8]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[9]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['Honk_duration','RMS','Intersection density','WiFi density']
X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[10]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['RMS','Intersection density','WiFi density']
X=list[['RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
clf = DecisionTreeClassifier(max_depth=8,min_samples_leaf=8)

# Fit the classifier to the training data
clf.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[11]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# Instantiate a k-NN classifier: knn

list=pd.read_csv('dataset.csv')
x=['RMS','Intersection density','WiFi density']
X=list[['RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
Y=list['Class'].values
Y_d=pd.DataFrame(Y)
Y_d_2=pd.get_dummies(Y_d)
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[16]:


import pandas as pd
import seaborn as sns

list=pd.read_csv('dataset.csv')
plt.figure()
sns.factorplot(x='WiFi count',col='Timelevel',data=list,kind='count')
plt.show()


# In[17]:


import pandas as pd
import seaborn as sns


list=pd.read_csv('dataset.csv')
plt.figure()
sns.scatterplot(x='RMS',y='Timelevel',data=list)
plt.show()


# In[18]:


import pandas as pd
import seaborn as sns


list=pd.read_csv('dataset.csv')
plt.figure()
sns.scatterplot(x='Intersection count',y='RMS',data=list)
plt.show()


# In[19]:


import pandas as pd
import seaborn as sns


list=pd.read_csv('dataset.csv')
plt.figure()
sns.scatterplot(x='Intersection count',y='Class',data=list)
plt.show()


# In[22]:


import pandas as pd
import seaborn as sns


list=pd.read_csv('dataset.csv')
list2=list.replace({'market':0,'normal_city':1,'highway':2,'slow':0,'normal':1,'fast':2,'very fast':3})
plt.figure()
sns.barplot(x='Zone',y='Class',data=list2)
plt.show()


# In[31]:


import pandas as pd
import seaborn as sns


list=pd.read_csv('6mar.csv')

def f(row):
    
    if row['WiFi count']<20:
        val=0
    if row['WiFi count']>=20 and row['WiFi count']<50:
        val=1
    if row['WiFi count']>=50:
        val=2
    return val
list['WiFi range']=list.apply(f
                              ,axis=1)
list.head()


# In[1]:


from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('bike_SOUND_2018_06_13_17_04_38_324.wav')
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, data[:,0], data[:,1], color='k') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('plot.png', dpi=100)
plt.show() 


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('bike_SOUND_2018_06_13_17_04_38_324.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if spf.getnchannels() == 2:
    #print('Just mono files'
    sys.exit(0)


plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


spf = wave.open('bike_SOUND_2018_06_13_17_04_38_324.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()

#If Stereo
if spf.getnchannels() == 2:
    #print('Just mono files'
    sys.exit(0)
Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()


# In[11]:


import pandas as pd
#X_names=['RMS','Intersection count','WiFi count','Zone','Honk_duration']
df=pd.read_csv('dataset.csv')
#Y_names=['Road_surface','RSI class','Honk_duration','Mean_speed_kmph']
df2=pd.read_csv('6mar.csv')

X=df[['RMS','Intersection count','WiFi count','Zone','Honk_duration']].values
X_d=pd.DataFrame(X)

Y=df2[['Road_surface','RSI class','Honk_duration','Mean_speed_kmph']].values
Y_d=pd.DataFrame(Y)

df_main=pd.concat([X_d,Y_d],axis=1)

print(df_main.head)


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns

#df=pd.read_csv('dataset.csv')

df2=pd.read_csv('6mar.csv')

#X=df[['RMS','Intersection count','WiFi count','Zone','Honk_duration']].values
#X_d=pd.DataFrame(X)

Y=df2[['Road_surface','RSI class','Honk_duration','WiFi density','Inter_count','Mean_speed_kmph']].values
Y_d=pd.DataFrame(Y)

#df_main=pd.concat([X_d,Y_d],axis=1)

#print(df_main.head)
df2.boxplot(by='Road_surface',column=['Mean_speed_kmph'],grid=False)


# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='RSI class',y='Mean_speed_kmph',data=df)


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='RSI class',y='SpeedRange',data=df)


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='RSI class',y='Mean_speed_kmph',data=df)


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='Honk_duration',y='Mean_speed_kmph',data=df)


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='Inter_count',y='Mean_speed_kmph',data=df)


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')


#sns.boxplot(x='Inter_count',y='Mean_speed_kmph',hue='Timelevel',data=df)
#%matplotlib qt
df[(df.Zone=='market') & (df.Inter_count<4)].boxplot(column=['Mean_speed_kmph'], by=['Timelevel', 'Zone','Inter_count'],figsize=(15,10))


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='RSI class',y='Mean_speed_kmph',hue='Timelevel',data=df)


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='RSI class',y='Mean_speed_kmph',hue='Zone',data=df)


# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')

sns.boxplot(x='Zone',y='Mean_speed_kmph',hue='Timelevel',data=df)


# In[5]:


# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')


sns.boxplot(x='Inter_count',y='Mean_speed_kmph',hue='Timelevel',data=df)


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')


#sns.boxplot(x='Inter_count',y='Mean_speed_kmph',hue='Timelevel',data=df)
#%matplotlib qt
df[(df.Zone=='highway')].boxplot(column=['Mean_speed_kmph'], by=['Timelevel', 'Zone','Inter_count'],figsize=(15,10))


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('6mar.csv')


#sns.boxplot(x='Inter_count',y='Mean_speed_kmph',hue='Timelevel',data=df)
#%matplotlib qt
df[(df.Zone=='market') & (df.Inter_count>3)].boxplot(column=['Mean_speed_kmph'], by=['Timelevel', 'Zone','Inter_count'],figsize=(15,10))


# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('6mar.csv')

honk_l=df[['Honk_duration','Mean_speed_kmph','Timelevel']].values
honk_d=pd.DataFrame(honk_l)

honk_d.loc[honk_d[0]<=4,'Honk_bin']=1
honk_d.loc[(honk_d[0]>4) &(honk_d[0]<=12),'Honk_bin']=2
honk_d.loc[(honk_d[0]>12) &(honk_d[0]<=20),'Honk_bin']=3
honk_d.loc[(honk_d[0]>20) &(honk_d[0]<=28),'Honk_bin']=4
honk_d.loc[(honk_d[0]>28),'Honk_bin']=5

sns.boxplot(x='Honk_bin',y=1,hue=2,data=honk_d)
#print(honk_d)


# In[48]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('6mar.csv')

honk_l=df[['Honk_duration','Mean_speed_kmph','Timelevel','Zone']].values
honk_d=pd.DataFrame(honk_l)

honk_d.loc[honk_d[0]<=4,'Honk_bin']='1'
honk_d.loc[(honk_d[0]>4) &(honk_d[0]<=12),'Honk_bin']='2'
honk_d.loc[(honk_d[0]>12) &(honk_d[0]<=20),'Honk_bin']='3'
honk_d.loc[(honk_d[0]>20) &(honk_d[0]<=28),'Honk_bin']='4'
honk_d.loc[(honk_d[0]>28),'Honk_bin']='5'
honk_d_f=pd.DataFrame(honk_d)
#print(honk_d_f)   #1->mean_speed 2->timelevel 3->zone
honk_d_f[(honk_d_f.Honk_bin<'2') ].boxplot(column=[1], by=[2, 3,'Honk_bin'],figsize=(20,12),fontsize='large')


# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('6mar.csv')

honk_l=df[['Honk_duration','Mean_speed_kmph','Timelevel','Zone']].values
honk_d=pd.DataFrame(honk_l)

honk_d.loc[honk_d[0]<=4,'Honk_bin']='1'
honk_d.loc[(honk_d[0]>4) &(honk_d[0]<=12),'Honk_bin']='2'
honk_d.loc[(honk_d[0]>12) &(honk_d[0]<=20),'Honk_bin']='3'
honk_d.loc[(honk_d[0]>20) &(honk_d[0]<=28),'Honk_bin']='4'
honk_d.loc[(honk_d[0]>28),'Honk_bin']='5'
honk_d_f=pd.DataFrame(honk_d)
#print(honk_d_f)      #1->mean_speed 2->timelevel 3->zone
honk_d_f[(honk_d_f.Honk_bin>'2') ].boxplot(column=[1], by=[2, 3,'Honk_bin'],figsize=(20,12),fontsize='large')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('6mar.csv')

honk_l=df[['Honk_duration','Mean_speed_kmph','Timelevel','Zone']].values
honk_d=pd.DataFrame(honk_l)

honk_d.loc[honk_d[0]<=4,'Honk_bin']='1'
honk_d.loc[(honk_d[0]>4) &(honk_d[0]<=12),'Honk_bin']='2'
honk_d.loc[(honk_d[0]>12) &(honk_d[0]<=20),'Honk_bin']='3'
honk_d.loc[(honk_d[0]>20) &(honk_d[0]<=28),'Honk_bin']='4'
honk_d.loc[(honk_d[0]>28),'Honk_bin']='5'
honk_d_f=pd.DataFrame(honk_d)

