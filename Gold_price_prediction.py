import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
#data ni preprocess cheyyali using pandas
gold_data=pd.read_csv('/content/gld_price_data.csv')
#print first five rows to know the data
gold_data.head()
gold_data.tail()
gold_data.shape
#idhi enduku ante asalu data entha umdi ani telusuodaniki
gold_data.info()
#idhi enduku ante manaki data lo enni datapoints telusukovadam kosam
#getting statistical measures of the data
gold_data.describe()
correlation=gold_data.corr()
#heatmap ni construct chestham to understand the corelation values
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
#checking the distribution of the gold price
sns.distplot(gold_data['GLD'])
#splitting the features and target
X=gold_data.drop(['Date','GLD'],axis=1)
Y=gold_data['GLD']
print(X,Y)
#splitting to train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)#model training by random forest regression
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)
print(X_test)
#model evaluation predction chestham
#input data ememi istham ante SPX ante stock prizes USO values and silver rate and usd value from the data set
input=(1447.160034,78.470001,15.18,1.471692)
inp=np.asarray(input)
inpu=inp.reshape(1,-1)
prediction=regressor.predict(inpu)
print("The prize of Gold is",prediction[0]*82)
#R squared error
error_score = metrics.r2_score(Y_test,prediction)
print("R squared error",error_score)
#compare the actual values and predicted value through plot
Y_test=list(Y_test)
plt.plot(Y_test,color='blue',label='Acutal Value')
plt.plot(prediction,color='green',label='Predicted value')
plt.title('Actual Price Vs Predicted value')
plt.xlabel('Number of values')
plt.ylabel('Gold price')
plt.legend()
plt.show()
