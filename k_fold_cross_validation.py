from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

dataset = load_wine()
x = dataset['data']
y = dataset['target']

kfold = KFold(n_splits=5, random_state=101, shuffle=True)

svm_acc, tree_acc = [], []
for train_i, test_i in kfold.split(x):
    x_train, x_test = x[train_i], x[test_i]
    y_train, y_test = y[train_i], y[test_i]

    svm = SVC(C=1.0)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    svm_acc.append(accuracy_score(y_test, y_pred))

    tree = DecisionTreeClassifier(ccp_alpha=0.2)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    tree_acc.append(accuracy_score(y_test, y_pred))

print(f'svm accuracy: {sum(svm_acc)/len(svm_acc)}')
print(f'decision tree accuracy: {sum(tree_acc)/len(tree_acc)}')
