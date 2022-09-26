from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report

dataset = load_wine()
x = dataset['data']
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101, stratify=y)

params = {'C': [0.01, 0.05, 0.2, 0.5, 0.8, 1.0], 'gamma': ['scale', 0.2, 0.5, 1.0], 'kernel': ['linear', 'rbf']}
cv = GridSearchCV(SVC(), param_grid=params, scoring='accuracy', cv=7, n_jobs=-1)
cv.fit(x_train, y_train)
print(cv.best_params_)
print(classification_report(y_test, cv.predict(x_test)))

params = {'max_depth': [5, 10, 20, 50, 100], 'ccp_alpha': [0.0, 0.2, 0.5, 0.8, 1.0, 2.0]}
cv = GridSearchCV(DecisionTreeClassifier(), param_grid=params, scoring='accuracy', cv=7, n_jobs=-1)
cv.fit(x_train, y_train)
print(cv.best_params_)
print(classification_report(y_test, cv.predict(x_test)))