# Machine Learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Utilities
import time

train = [
    "crayons can be black",
    "crayons can be white",
    "vintage photographs are black and white",
    "markers can be black",
    "markers can be white"
]

labels = [
    ['black'],
    ['white'],
    ['black', 'white'],
    ['black'],
    ['white']
]

test = [
    'ink can be black',
    'ink can be white',
    'checkers are both black and white'
]

pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', OneVsRestClassifier(LinearSVC()))
    # ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# binarize
binarizer = MultiLabelBinarizer()
labels = binarizer.fit_transform(labels)
classes = binarizer.classes_

# train & predict
pipe.fit(train, labels)
predicted = pipe.predict(test)

# display
for item, labels in zip(test, predicted):
    labels = [classes[i] for i, label in enumerate(labels) if label]
    print(item + ':', labels)

# benchmark
benchmark = [y for x in range(100000) for y in test]
start = time.time()
predicted = pipe.predict(benchmark)
print('Prediction Completed:', time.time() - start, 'second(s)')
