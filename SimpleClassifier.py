from sklearn import tree
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier


clf = tree.DecisionTreeClassifier() #Decisiontree
clf1 = GaussianNB() 
clf2 = SVC()
clf3 = GaussianProcessClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

prediction = clf.predict([[190,70,43]])
prediction1 = clf.predict([[190,70,43]])
prediction2 = clf.predict([[190,70,43]])
prediction3 = clf.predict([[190,70,43]])

print("Dicision tree prediction", prediction)
print("GaussianNB", prediction1)
print("SVC", prediction2)
print("GaussianProcess", prediction3)

"""
Test Example. 
from sklearn import tree

clf = tree.DecisionTreeClassifier() 

features = [[50 , 0 ] , [ 80, 0 ] , [120 , 1 ] , [160, 1] , [180 , 2] , [200 ,2] ] 

lables = ['honda' , 'honda' , 'audi' , 'audi' , 'ferrari', 'ferrai']

clf = clf.fit(features, lables) 

prediction = clf.predict([[90, 1]]) 

print (prediction)


"""
