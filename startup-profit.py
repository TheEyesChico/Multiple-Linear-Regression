import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv')
# f=open('data.pickle','wb')
# pickle.dump(df,f)
# f.close()

pickle_in = open('data.pickle',"rb")
df = pickle.load(pickle_in)

# for i in ['R&D Spend','Administration','Marketing Spend']:
#     sns.scatterplot(i,'Profit',data=df,hue='State')
#     plt.show()

y=df.loc[:,"Profit"]

state=pd.get_dummies(df['State'],drop_first=True)
df.drop(['State','Profit'],axis=1, inplace=True)

X=pd.concat([df,state],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

reg = LinearRegression().fit(X_train, y_train)
# print(np.array(y_test),"\n")

y_pred=reg.predict(X_test)
# print(y_pred,"\n")

score=r2_score(y_test,y_pred)
print(score)