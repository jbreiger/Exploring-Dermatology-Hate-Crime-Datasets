# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:42:35 2021

@author: Josh
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.linear_model import LinearRegression




#New_Derma = pd.read_csv('dermatology.csv',sep='\s+',header=None)

Derma = pd.read_excel('new_dermatology.xlsx')


Derma['Age'] = pd.to_numeric(Derma.Age, errors='coerce')

#Derma = Derma.dropna(subset= ['Age'])

Derma = Derma.dropna()


#derm_data = pd.read_table('derm)

#Let’s try determining the type of disease based on the patient’s Age. Use gradient descent (GD) 
#to build your regression model (model1). Start by writing the GD algorithm
# and then implement it using a programming language of your choice. [10 points]


X = Derma[['Age']]

X = sm.add_constant(X) # Adds constant/intercept term
y = Derma[['Disease']]

plt.scatter(Derma.Age, Derma.Disease)
plt.xlabel = ("Age")
plt.ylabel = ("Disease")
plt.title('Age VS Disease')
plt.show()

lr_model = sm.OLS(y,X).fit()

def grad_descent(X, y, alpha, epsilon):
    X = X + np.array([1,0])
    iteration = [0]
    i = 0
    theta = np.ones(shape= (X.shape[1], 1))
    #shape=(X.shape[1],1)
    cost = [np.transpose(X @ theta - y) @ (X @ theta - y)]
    delta = 1
    while (delta>epsilon):
        theta = theta - alpha*((np.transpose(X)) @ (X @ theta - y))
        cost_val = (np.transpose(X @ theta - y)) @ (X @ theta - y)
        cost.append(cost_val)
        #print("cost is", cost)
        delta = abs(cost[i+1]-cost[i])
        if ((cost[i+1]-cost[i]) > 0):
            print("The cost is increasing. Try reducing alpha.")
            break
        iteration.append(i)
        i += 1
        
    print("Completed in %d iterations." %(i))
    return(theta)



y = y.to_numpy()

X = preprocessing.scale(X)

theta = grad_descent(X = preprocessing.scale(X), y=y, alpha=0.0001, epsilon = 10**-10)

print (theta)

#first one is intercept


#Use random forest on the clinical as well as histopathological attributes to classify
# the disease type (model2). [5 points]

#moved age to column 1 of the Excel document to able to group them together better

clinical_X = Derma.iloc[:,0:12]

clinical_X = preprocessing.scale(clinical_X)

y = Derma[['Disease']]

accuracy_list = list()


for x in range(200):
    X_train, X_test, y_train, y_test = train_test_split(clinical_X, y, test_size = 0.3)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, predictions))

avg_accuracy = sum(accuracy_list)/len(accuracy_list)

#print(confusion_matrix(y_test, predictions))
#print(accuracy_score(y_test, predictions))

#Do the same thing with Histopathological variables 


histo_X = Derma.iloc[:, 12:34]

histo_X = preprocessing.scale(histo_X)

for x in range(200):
    X_train, X_test, y_train, y_test = train_test_split(histo_X, y, test_size = 0.3)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    predictions = dtree.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, predictions))

avg_accuracy = sum(accuracy_list)/len(accuracy_list)



print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

#3


#use the same histoX and ClinicalX as before

clinical_X_normalized = clinical_X.apply(lambda x: (x-min(x))/(max(x)-min(x)))

histo_X_normalized = histo_X.apply(lambda x: (x-min(x))/(max(x)-min(x))) 

accuracy_list = list()
total_accuracy = list()

#Histo data

for x in range(200):

    X_train, X_test, y_train, y_test = train_test_split(clinical_X_normalized, y, test_size= 0.30)
    knn = KNeighborsClassifier(n_neighbors= 10)

    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    accuracy_list.append(accuracy_score(y_test, predictions))
    
avg_accuracy = sum(accuracy_list)/len(accuracy_list)




#4 clustering techniques

#clinical X and histo_X used over from before

#new_X = Derma.drop(['Disease'], axis=1)

scaler = StandardScaler()
scaler.fit_transform(Derma)

#kmeans = KMeans(n_clusters=4)

kmeans = KMeans(n_clusters=6)
y_means = kmeans.fit_predict(Derma)

centroids = kmeans.cluster_centers_
print(centroids)
print(y_means)

plt.figure(figsize=(10,10))
plt.title('Divisive clustering with k-means')
plt.scatter(Derma['Disease'], Derma['Scathing'], c=y_means, cmap='rainbow')
plt.scatter(centroids[:,1], centroids[:,2], c='black',s=100)
plt.xlabel('Disease')
plt.ylabel('Scathing')
plt.show()

plt.figure(figsize=(10,10))
plt.title('Agglomerative clustering')

Dendrogram = sch.dendrogram((sch.linkage(Derma, method='ward')))




#2 part A

Hate = pd.read_csv('hatecrime_updated.csv')

#plt.scatter(Hate.gini_index, Hate.hate_crimes_per_100k_splc)


#How does income inequality relate to the number of hate crimes and hate incidents? [5 points]


hate_y= Hate[['avg_hatecrimes_per_100k_fbi']]

#iteration 1
hate_X = Hate[['share_unemployed_seasonal', 'share_white_poverty', 'gini_index',"median_household_income"]] #maybe dont include gini

#iteration 2
hate_X = Hate[['share_unemployed_seasonal', 'share_white_poverty', 'gini_index']] #maybe dont include gini

#iteration 3

hate_X = Hate[[ 'share_white_poverty', 'gini_index']]

hate_X = sm.add_constant(hate_X)

lr_model = sm.OLS(hate_y,hate_X).fit()


lr_model = sm.OLS(hate_y,hate_X).fit()
print(lr_model.summary())

hate_y = Hate[["hate_crimes_per_100k_splc"]]

hate_X = Hate[['share_unemployed_seasonal', 'share_white_poverty', 'gini_index',"median_household_income"]] 

hate_X = Hate[['share_white_poverty', 'gini_index']] 

hate_X = sm.add_constant(hate_X)

print(lr_model.summary())

plt.scatter(Hate.gini_index, Hate.hate_crimes_per_100k_splc)


#2 Part B

#Non-FBI



race_hate_X = Hate[['share_non_white', 'share_population_with_high_school_degree', 'share_non_citizen', 'share_voters_voted_trump']] #iteration 1
race_hate_X = Hate[[ 'share_population_with_high_school_degree','share_voters_voted_trump']] 

race_hate_X = Hate[[ 'share_non_citizen','share_voters_voted_trump']] 

race_hate_X = sm.add_constant(race_hate_X)

hate_y= Hate[['hate_crimes_per_100k_splc']]

lr_model = sm.OLS(hate_y,race_hate_X).fit()
print(lr_model.summary())

#FBI

race_hate_X = Hate[['share_non_white', 'share_population_with_high_school_degree', 'share_non_citizen', 'share_voters_voted_trump']] #iteration 1
race_hate_X = Hate[[ 'share_population_with_high_school_degree', 'share_non_citizen']] 


race_hate_X = sm.add_constant(race_hate_X)

hate_y= Hate[['avg_hatecrimes_per_100k_fbi']]

lr_model = sm.OLS(hate_y,race_hate_X).fit()
print(lr_model.summary())


#3 part c 
#How does the number of hate crimes vary across states?
#Is there any similarity in number of hate incidents (per 100,000 people) between some states than in others 

import seaborn as sns

state_X = Hate

corrMatrix = state_X.corr()


corrMatrix = state_X.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

state_X = Hate[['hate_crimes_per_100k_splc', 'avg_hatecrimes_per_100k_fbi']]
scaler = StandardScaler()
testing= scaler.fit_transform(state_X)
testing = pd.DataFrame(testing)
testing.columns= ["FBI", "SPLC"]
boxplot = testing.boxplot(column = ["FBI", "SPLC"])



ax = sns.stripplot(x=Hate["hate_crimes_per_100k_splc"])
ax = sns.stripplot(x=Hate["avg_hatecrimes_per_100k_fbi"])




#EXTRA NOTES 

#can use clustering by state 

#state_X = Hate.loc[:, [Hate.columns != 'state', 'avg_hatecrimes_per_100k_fbi]]

#state_X = Hate[['median_household_income','share_unemployed_seasonal','share_non_white','share_population_in_metro_areas', 'share_population_with_high_school_degree', 'share_non_citizen', 'share_white_poverty', 'gini_index', 'share_non_white', 'share_voters_voted_trump', 'hate_crimes_per_100k_splc']]
#
#state_X = Hate[['hate_crimes_per_100k_splc', 'avg_hatecrimes_per_100k_fbi']]
#
#scaler = StandardScaler()
#testing= scaler.fit_transform(state_X)
#
#kmeans = KMeans(n_clusters=4)
#y_means = kmeans.fit_predict(state_X)
#
#centroids = kmeans.cluster_centers_
#print(centroids)
#print(y_means)
#
#plt.figure(figsize=(10,10))
#plt.title('Divisive clustering with k-means')
#plt.scatter(state_X['share_voters_voted_trump'], state_X['hate_crimes_per_100k_splc'], c=y_means, cmap='rainbow')
#plt.scatter(centroids[:,1], centroids[:,2], c='black',s=100)
