from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

import utils

print('Loading data...')
features = utils.load_json_by_line('data/features.json')
x = [f['repr'] for f in features]
y = [f['label'] for f in features]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

clf = svm.LinearSVC()

print('Fitting')
clf.fit(x_train, y_train)

print('Predicting')
preds = clf.predict(x_test)
acc = metrics.accuracy_score(preds, y_test)

print('accuracy:', acc)
