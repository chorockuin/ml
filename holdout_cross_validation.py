from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, plot_roc_curve, roc_auc_score, classification_report

def print_validation(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print('accuracy:', accuracy_score(y_test, y_pred))
    print('f1 scores:', f1_score(y_test, y_pred, average='macro'))

dataset = load_wine()
x = dataset['data']
y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=101, stratify=y)

svm = SVC(C=1.0)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print_validation(y_test, y_pred)

tree = DecisionTreeClassifier(ccp_alpha=0.2)
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
print_validation(y_test, y_pred)
