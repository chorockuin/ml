from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from scipy.stats import uniform, randint

dataset = load_wine()
x = dataset['data']
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101, stratify=y)

param_dists = {'C': uniform(loc=0, scale=1.0), 'gamma': uniform(loc=0, scale=1.0), 'kernel': ['linear', 'rbf']}
cv = RandomizedSearchCV(SVC(), param_dists, scoring='accuracy', n_iter=1000, cv=7, n_jobs=-1)
cv.fit(x_train, y_train)
print(cv.best_params_)
print(classification_report(y_test, cv.predict(x_test)))

param_dists = {'max_depth': randint(1,101), 'ccp_alpha': uniform(loc=0.0, scale=10.0)}
cv = RandomizedSearchCV(DecisionTreeClassifier(), param_dists, n_iter=1000, scoring='accuracy', cv=7, n_jobs=-1)
cv.fit(x_train, y_train)
print(cv.best_params_)
print(classification_report(y_test, cv.predict(x_test)))
