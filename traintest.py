"""
Alex Todorovic CMPS-3240
Using pandas for data processing, LinearSVC from sklearn.svm for training and testing, metrics from sklearn for evaluating accuracy

Note:
Although I planned on using several seasons of games for training I've only included one for training and one for testing
in this program. This was mainly for simplicity and ease of use as several .csv files include NaN values and values that
are not processable that cause errors. In the future I plan on adding multiple seasons of games for training and testing,
but I must first find and clean the .csv files causing problems.

This model still yields a surprisingly high accuracy score with just 380 games to train on.
I've gotten as low as 0.45 and as high as 0.63.
"""
from sklearn import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  GradientBoostingClassifier



'''Training data processing'''
league = pd.read_csv('E0-3.csv', header=None,
                                                #these are the names of the columns
                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])


'''Features list: home shots, away shots, home shots on target, away shots on target,
home fouls, away fouls, home corners, away corners, home yellow cards, away yellow
cards, home red cards, away red cards'''

#features = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',]
features = ['HST', 'AST','HTHG','HTAG']
x_train = league[features]
x_train = x_train[1:] #removing labels
y_train = league[['FTR']]
y_train = y_train[1:] #removing labels


'''     Testing data processing     '''
league = pd.read_csv('E0-2.csv', header=None,

                         names=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                                'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                                'bet1', 'bet2', 'bet3', 'bet4', 'bet5', 'bet6', 'bet7', 'bet8', 'bet9', 'bet10', 'bet11',
                                'bet12', 'bet13', 'bet14', 'bet15', 'bet16', 'bet17', 'bet18', 'bet19', 'bet20', 'bet21',
                                'bet22', 'bet23','bet24', 'bet25', 'bet26', 'bet27', 'bet28', 'bet29', 'bet30', 'bet31',
                                'bet32','bet33', 'bet34', 'bet35', 'bet36', 'bet37', 'bet38', 'bet39', 'bet40', 'bet41',
                                'bet42'])

'''Doing the same thing in training data processing, except I'm using a different .csv file'''
x_test = league[features]
x_test = x_test[1:]
y_test = league[['FTR']]
y_test = y_test[1:]


n = 50
accuracy = 0
precision = 0
recall = 0
f1 = 0

svm1 = LinearSVC()
for i in range(n):

    svm1.fit(x_train, y_train)
    y_pred = svm1.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Linear SVC~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')


accuracy = 0
precision = 0
recall = 0
f1 = 0
clf = linear_model.SGDClassifier()
for i in range(n):

    clf.fit(x_train, y_train)
    y_pred1 = clf.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred1)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred1, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred1, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred1, average='macro')                 #f1

print('~~~Evaluating SGD Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')

accuracy = 0
precision = 0
recall = 0
f1 = 0
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
for i in range(n):
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating 5 nearest neighbors Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')

accuracy = 0
precision = 0
recall = 0
f1 = 0

accuracy = 0
precision = 0
recall = 0
f1 = 0
lr = LogisticRegression()
for i in range(n):
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Logistic Regression Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')






accuracy = 0
precision = 0
recall = 0
f1 = 0
tree = tree.DecisionTreeClassifier(criterion='gini')
for i in range(n):
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Decision Tree Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')

accuracy = 0
precision = 0
recall = 0
f1 = 0
gaus = GaussianNB()
for i in range(n):
    gaus.fit(x_train, y_train)
    y_pred = gaus.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Gaussian NB Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')

accuracy = 0
precision = 0
recall = 0
f1 = 0
booster= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
for i in range(n):
    booster.fit(x_train, y_train)
    y_pred = booster.predict(x_test)

    accuracy += metrics.accuracy_score(y_test, y_pred)                      #accuracy
    precision += metrics.precision_score(y_test, y_pred, average='macro')   #precision
    recall += metrics.recall_score(y_test, y_pred, average='macro')         #recall
    f1 += metrics.f1_score(y_test, y_pred, average='macro')                 #f1

print('~~~Evaluating Gradient Boosting Classifier~~~')
print('Accuracy score: ' + str(accuracy / n) + '\n')
print('Precision score: ' + str(precision / n) + '\n')
print('Recall score: ' + str(recall / n) + '\n')
print('F1 score: ' + str(f1 / n) + '\n')