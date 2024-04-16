import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


df = pd.read_csv('DataSets.csv')
df
df.columns
df.rename(columns = {'Response':'label','Vehicle_Age':'VAge'},inplace = True)
def apply_results(label):
    if(label==0 ):
        return 0 # Not Interested
    elif(label==1 ):
        return 1 # Interested

df['results'] = df['label'].apply(apply_results)
df.drop(['label'],axis = 1, inplace = True)
results = df['results'].value_counts()

cv = CountVectorizer()
X = df['VAge']
y = df['results']


print("VAge")
print(X)
print("Results")
print(y)

X = cv.fit_transform(X)

models = []
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.20)
X_train.shape,X_test.shape,y_train.shape


print("Naive Bayes")

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)
predict_nb = NB.predict(X_test)
naivebayes = accuracy_score(y_test, predict_nb) * 100
print(naivebayes)
print(confusion_matrix(y_test,predict_nb))
print(classification_report(y_test, predict_nb))
models.append(('naive_bayes', NB))

# SVM Model
print("SVM")
from sklearn import svm
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
predict_svm = lin_clf.predict(X_test)
svm_acc = accuracy_score(y_test, predict_svm) * 100
print(svm_acc)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, predict_svm))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predict_svm))
models.append(('svm', lin_clf))

print("Logistic Regression")

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("ACCURACY")
print(accuracy_score(y_test, y_pred) * 100)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
models.append(('logistic', reg))



print("Decision Tree Classifier")
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
pred_dt = DT.predict(X_test)
DT.score(X_test, y_test)
print("ACCURACY")
print(accuracy_score(y_test, pred_dt) * 100)
print("CLASSIFICATION REPORT")
print(classification_report(y_test, pred_dt))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, pred_dt))

models.append(('DecisionTreeClassifier', DT))


predicts = 'Predictions.csv'
df.to_csv(predicts, index=False)
df.to_markdown
