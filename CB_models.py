# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib

# %%
df=pd.read_csv("weatherAUS.csv")

pd.set_option("display.max_columns", None)
df


# %%
num_vars= [parameter for parameter in df.columns if df[parameter].dtypes !='O']
dis_vars=[parameter for parameter in num_vars if len(df[parameter].unique())<25]
cont_vars=[parameter for parameter in num_vars if parameter not in dis_vars]
cat_vars=[parameter for parameter in df.columns if parameter not in num_vars]

# %%
print("num_vars.count is {}".format(len(num_vars)))
print("dis_vars.count is {}".format(len(dis_vars)))
print("cont_vars.count is {}".format(len(cont_vars)))
print("cat_vars.count is {}".format(len(cat_vars)))

# %%
#len(df['WindGustDir'].unique())

# %%
df.isnull().sum()*100/len(df)
# this gives the percentage of missing values in each column


# %%
for col in num_vars:
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Impute missing values in categorical columns with mode
for col in cat_vars:
        df[col].fillna(df[col].mode()[0], inplace=True)

# %%
df.to_csv('app.csv',index=False)

# %%
print(num_vars)

# %%
#creating a function to handle missinng values of column by imputing them
def randomsampleimputation(df,variable):
    df[variable]=df[variable]
    rand_samp=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    rand_samp.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable]=rand_samp

# %%
randomsampleimputation(df, "Cloud9am")
randomsampleimputation(df, "Cloud3pm")
randomsampleimputation(df, "Evaporation")
randomsampleimputation(df, "Sunshine")

# %%
df.isnull().sum()

# %%

corrmat=df.corr(method="spearman",numeric_only=True)
plt.figure(figsize=(20,20))
heat_map=sns.heatmap(corrmat,annot=True)

# %%

for feature in cont_vars:
    data=df.copy()
    
    sns.displot(df[feature])
    
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.figure(figsize=(15,15))
    plt.show()



# %%
for feature in cont_vars:
    data=df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))

# %%
for feature in cont_vars:
    if(df[feature].isnull().sum()*100/len(df)>0):
        df[feature]=df[feature].fillna(df[feature].median())
        #df[feature].fillna(df[feature].median(),inplace=True) same as above line (using inplace)
    

# %%
df.isnull().sum()*100/len(df)

# %%
dis_vars

# %%
def mode_nan(df,variable):
    mode=df[variable].value_counts().index[0]
    df[variable].fillna(mode,inplace=True)
mode_nan(df,"Cloud9am")
mode_nan(df,"Cloud3pm")
    

# %%
#categoriacal variables
df["RainToday"]=pd.get_dummies(df["RainToday"],drop_first=True)
df["RainTomorrow"]=pd.get_dummies(df["RainTomorrow"],drop_first=True)

# %%
#value_1_present = (df['RainToday'] == 1).any()
#value_2_present =(df['RainTomorrow'] == 1).any()
#print(value_1_present)
#print(value_2_present)
# checking for the presence of a specific value in columns

# %%
#grouping all unique values in categorical columns based on the mean of corresponding values in RainTomorrow column
for feature in cat_vars:
    print(feature,(df.groupby([feature])["RainTomorrow"].mean().sort_values(ascending=False)).index)
    
    

# %%
windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
       'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
       'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
       'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15} 

#mapping categoriacal values to numerical values for computation

df["WindGustDir"]=df["WindGustDir"].map(windgustdir)
df["WindDir9am"]=df["WindDir9am"].map(winddir9am)
df["WindDir3pm"]=df["WindDir3pm"].map(winddir3pm)

# %%
df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])
df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])

#df["WindDir9am"]=df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
#df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0],inplace=True)
# either use inplace=True or assigment operator to change the dataframe and fill missing value using fillna() method

# %%
df.isnull().sum()*100/len(df)

# %%
#df1=df.groupby(["Location"])["RainTomorrow"].value_counts(sort=False).unstack()
#both the above and below line do the same operation
df1=df.groupby(["Location"])["RainTomorrow"].value_counts().sort_values().unstack()

# %%
df1

# %%
df1.iloc[:,1].sort_values(ascending=False).index

# %%
df1.head()

# %%
df1.iloc[:, 1].sort_values(ascending=False).index

# %%
len(df1.iloc[:,1].sort_values(ascending=False).index)

# %%


# %%
location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}
df["Location"]=df["Location"].map(location)

# %%
df["Date"]=pd.to_datetime(df["Date"],format="%Y-%m-%dT" ,errors="coerce")

# %%
df["Date-month"]=df["Date"].dt.month
df["Date-day"]=df["Date"].dt.day
df

# %%
#df.corr()
corrmat=df.corr()
plt.figure(figsize=(20,20))
heat_map=sns.heatmap(corrmat,annot=True,linewidth=0.5)

# %%
#sns.countplot(df["RainTomorrow"])

# %%
for feature in cont_vars:
    data=df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))
    

# %%
for feature in cont_vars:
    print(feature)

# %%
"""
# MinTemp Column
"""

# %%
#another way to do what above is done is to use if condition or loop as shown below
 #for index, row in df.iterrows():
#    if row["MinTemp"] >= 30.45:
  #      df.at[index, "MinTemp"] = 30.45
  #  elif row["MinTemp"] <= -5.95:
  #      df.at[index, "MinTemp"] = -5.95 


# %%
# Now we are going to find outliers in every feature of cont_vars and replace them using IQR method

# %%
"""
# MaxTemp 
"""

# %%
"""
# Rainfall
"""

# %%
"""
# Function to replace outlier with IQR method
"""

# %%
# calculation upper and lower bound values and replacing ouliers with them using IQR method directly
'''IQR=df.MinTemp.quantile(0.75)-df.MinTemp.quantile(0.25)
lower_bridge=df.MinTemp.quantile(0.25)-(IQR*1.5)
upper_bridge=df.MinTemp.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)'''

# %%
    #You can find outliers using this IQR method and replace them with lower bound and upper bound values 
'''     Q1=df["MinTemp"].quantile(0.25)
            Q3=df["MinTemp"].quantile(0.75)
            IQR=Q3-Q1
            lower_bound=Q1-1.5*IQR
            upper_bound=Q3+1.5*IQR
            df.loc[df["MinTemp"] > lower_bound,"MinTemp"]=lower_bound
            df.loc[df["MinTemp"] > upper_bound,"MinTemp"]=upper_bound
            print(upper_bound,lower_bound) '''

# %%
# Instead of calculating IQR, lower bound and upper bound for every column,we replace it with a function and call it whenever necessary

# %%


def replace_outliers_with_bounds(df, cont_vars):
    for column in cont_vars:
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with lower and upper bounds
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound



replace_outliers_with_bounds(df, cont_vars)

# Print or handle the modified DataFrame
print(df)


# %%
for feature in cont_vars:
    data=df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))

# %%
for feature in cont_vars:
    print(feature)
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stats.probplot(df[feature],dist='norm', plot=plt)
    plt.show()

# %%
#df.to_csv("preprocessed_2.csv",index=False)

# %%
df

# %%

X=df.drop(["RainTomorrow","Date",'Date-month','Date-day'],axis=1)
Y=df["RainTomorrow"]

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state = 0)

# %%
from sklearn.impute import SimpleImputer
imp_X=SimpleImputer(strategy='median')
X_train_imp=imp_X.fit_transform(X_train)

X_train_imp=np.where(np.isinf(X_train_imp),np.nan,X_train_imp)
X_train_imp=imp_X.fit_transform(X_train_imp)

imp_Y=SimpleImputer(strategy='median')
Y_train_imp=imp_Y.fit_transform(Y_train.values.reshape(-1,1)).ravel()

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state = 0)

# %%
Y_train

# %%
sm=SMOTE(random_state=0)
X_train_res, Y_train_res=sm.fit_resample(X_train_imp,Y_train_imp)
print("no of class before fit{}".format(Counter(Y_train_imp)))
print("no of class after fit{}".format(Counter(Y_train_res)))

# %%
df.shape

# %%
cat=CatBoostClassifier(iterations=2000,verbose=200)
cat.fit(X_train, Y_train)                                        
Y_pred=cat.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(precision_score(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

# %%
#metrics.plot_roc_curve(cat, X_test, Y_test)
#metrics.roc_auc_score(Y_test, Y_pred, average=None)  #remove comment when necessary
joblib.dump(cat, "./models/cat.pkl")

# %%
rf=RandomForestClassifier()
rf.fit(X_train_res,Y_train_res)
#print("NaN or Infinite values in X_test: {}".format(np.isnan(X_test).sum().sum() + np.isinf(X_test).sum().sum()))

# %%
print("NaN or Infinite values in X_test: {}".format(np.isnan(X_test).sum().sum() + np.isinf(X_test).sum().sum()))

# %%
imp_X_test=SimpleImputer(strategy="median")
X_test_imp=imp_X_test.fit_transform(X_test)

# %%
Y_pred1=rf.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred1))
print(precision_score(Y_test,Y_pred1))
print(accuracy_score(Y_test,Y_pred1))
print(classification_report(Y_test,Y_pred1))
#metrics.plot_roc_curve(rf,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred1,average=None)
joblib.dump(rf, "./models/rf.pkl")

# %%
logreg=LogisticRegression()
logreg.fit(X_train_res,Y_train_res)
Y_pred2=logreg.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred2))
print(accuracy_score(Y_test,Y_pred2))
print(classification_report(Y_test,Y_pred2))
#metrics.plot_roc_curve(logreg,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred2,average=None)
joblib.dump(logreg, "./models/logreg.pkl")

# %%
gnb=GaussianNB()
gnb.fit(X_train_res,Y_train_res)
Y_pred3=gnb.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred3))
print(precision_score(Y_test,Y_pred3))
print(accuracy_score(Y_test,Y_pred3))
print(classification_report(Y_test,Y_pred3))
#metrics.plot_roc_curve(gnb,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred,average=None)
joblib.dump(gnb, "./models/gnb.pkl")

# %%
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_res,Y_train_res)
Y_pred4=knn.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred4))
print(accuracy_score(Y_test,Y_pred4))
print(classification_report(Y_test,Y_pred4))
#metrics.plot_roc_curve(knn,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred4)
joblib.dump(knn, "./models/knn.pkl")

# %%
xgb=XGBClassifier()
xgb.fit(X_train_res,Y_train_res)
Y_pred5=xgb.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred5))
print(accuracy_score(Y_test,Y_pred5))
print(classification_report(Y_test,Y_pred5))
#metrics.plot_roc_curve(xgb,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred5,average=None)
joblib.dump(xgb, "./models/xgb.pkl")

# %%
'''svc=SVC()
svc.fit(X_train_res,Y_train_res)
Y_pred6=svc.predict(X_test_imp)
print(confusion_matrix(Y_test,Y_pred6))
print(accuracy_score(Y_test,Y_pred6))
print(classification_report(Y_test,Y_pred6))
metrics.plot_roc_curve(svc,X_test_imp,Y_test)
metrics.roc_auc_score(Y_test,Y_pred5,average=None)'''


# %%
'''joblib.dump(cat, "./models/cat.pkl")
joblib.dump(rf, "./models/rf.pkl")
joblib.dump(logreg, "./models/logreg.pkl")
joblib.dump(gnb, "./models/gnb.pkl")
joblib.dump(knn, "./models/knn.pkl")
joblib.dump(xgb, "./models/xgb.pkl")
#joblib.dump(svc, "./models/svc.pkl")'''

# %%
df[(df['Evaporation']==2) & (df['Sunshine']==23)]


# %%
df.isnull().sum()

# %%


# %%
