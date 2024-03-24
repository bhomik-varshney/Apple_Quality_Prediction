import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('./apple_quality.csv')
print(data.head())

#Exploratory_Data_Analysis:
data_cols = data.columns
print(data_cols)
#we will check how these factors make impact on the quality of apple.
# print(data.isnull().sum())
data = data.dropna() #dropping null values.
print(data.isnull().sum())
print(data['Size'].describe())
# plt1 = px.scatter(data,x= 'Size',y ='Quality')
# plt1.show() #size doesnot play an important role in determining the QUALITY.
print(data.dtypes)
data.loc[(data['Quality']=='good','Quality')]=1
data.loc[(data['Quality']=='bad','Quality')]=0
data['Acidity'] = data['Acidity'].astype(float)
data['Quality'] = data['Quality'].astype(int)
print(data.dtypes)

f,ax= plt.subplots(1,2,figsize = (18,8))
y1 = sns.histplot(x='Weight',ax=ax[0],kde=True,data = data)
y2 = sns.histplot(x='Sweetness',ax=ax[1],kde = True,data = data)
plt.show()
#making a co relation graph of it.

# plt2 = sns.heatmap(data.corr(),cmap='RdYlGn',annot = True,annot_kws={'size':15},linewidths=0.2)
# plt.show()
#Size,Sweetness and juiciness are some factors which has important role, whereas ripeness has negative co-relation with Quality of apples.
#if the size of the apple is large,it will be less sweet.

#making a prediction models:-
import random
random.seed(42)
from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(data,test_size=0.2,random_state=42)
print(train_df.head())

input_cols = ['A_id', 'Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness',
       'Ripeness', 'Acidity']
target_cols = ['Quality']
train_inputs = train_df[input_cols]
train_targets= train_df[target_cols]
val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]

class dumb_model:
       def fit(self,input,target):
              pass
       def predict(self,input):
              return np.random.randint(2,size=len(input))

dumb_model = dumb_model()
dumb_model.fit(train_inputs,train_targets)
pred1 = dumb_model.predict(train_inputs)
pred2 = dumb_model.predict(val_inputs)
from sklearn.metrics import accuracy_score
a1 = accuracy_score(train_targets,pred1)
a2 = accuracy_score(val_targets,pred2)
print(a1,a2)  #around 50 percent our dumb model is accurate about its prediction

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(train_inputs,train_targets)
pred3 = model1.predict(train_inputs)
pred4 = model1.predict(val_inputs)
a3 = accuracy_score(train_targets,pred3)
a4 = accuracy_score(val_targets,pred4)
print(a3,a4)

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(max_depth=25)
model2.fit(train_inputs,train_targets)
pred5 = model2.predict(train_inputs)
pred6 = model2.predict(val_inputs)
a5= accuracy_score(train_targets,pred5)
a6 = accuracy_score(val_targets,pred6)
print(a5,a6)  #maximum val_accuracy is 90.5

import xgboost as xgb
model3 = xgb.XGBClassifier()
model3.fit(train_inputs,train_targets)
pred6 = model3.predict(train_inputs)
pred7 = model3.predict(val_inputs)
a6 = accuracy_score(train_targets,pred6)
a7 = accuracy_score(val_targets,pred7)
print(a6,a7)

from sklearn.ensemble import HistGradientBoostingClassifier
model4 = HistGradientBoostingClassifier()
model4.fit(train_inputs,train_targets)
pred8 = model4.predict(train_inputs)
pred9 = model4.predict(val_inputs)
a8 = accuracy_score(train_targets,pred8)
a9 = accuracy_score(val_targets,pred9)
print(a8,a9)

from sklearn.neighbors import KNeighborsClassifier
model5 = KNeighborsClassifier()
model5.fit(train_inputs,train_targets)
pred10 = model5.predict(train_inputs)
pred11 = model5.predict(val_inputs)
a10 = accuracy_score(train_targets,pred10)
a11 = accuracy_score(val_targets,pred11)
model5 = KNeighborsClassifier()
model5.fit(train_inputs,train_targets)
pred10 = model5.predict(train_inputs)
pred11 = model5.predict(val_inputs)
a10 = accuracy_score(train_targets,pred10)
a11 = accuracy_score(val_targets,pred11)  #KNN is giving very bad accuracy.
print(a10,a11)
from sklearn.svm import SVC
model6 = SVC()
model6.fit(train_inputs,train_targets)
pred12 = model6.predict(train_inputs)
pred13 = model5.predict(val_inputs)
a12 = accuracy_score(train_targets,pred12)
a13 = accuracy_score(val_targets,pred13)
print(a12,a13)

from sklearn.ensemble import VotingClassifier
model7 = VotingClassifier([('model4',model4),('model2',model2),('model3',model3)])
model7.fit(train_inputs,train_targets)
pred14 = model7.predict(train_inputs)
pred15 = model7.predict((val_inputs))
a14 = accuracy_score(train_targets,pred14)
a15 = accuracy_score(val_targets,pred15)
print(a14,a15) #maximum val accuracy is 91 percent whereas maximum train accuracy is 100 percent.



