import sklearn
import nltk
import scipy as sp


from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC


MLCOMPDIR = r'LOCATION OF CORPUS'

trainNews = load_mlcomp('16NepaliNews', 'train', mlcomp_root= MLCOMPDIR)
testNews = load_mlcomp('16NepaliNews', 'test', mlcomp_root= MLCOMPDIR)

''' Nepali Stop Words '''
# The stop words file is copied into the stopwords directory of nltk.data\corpora folder

stopWords = set(nltk.corpus.stopwords.words('nepali'))


''' Testing and Training Data '''
xTrain = trainNews.data
xTest = testNews.data
yTrain = trainNews.target
yTest = testNews.target


''' Vectorizer '''

tfidfVectorizer = TfidfVectorizer(tokenizer= lambda x: x.split(" "),
                                  sublinear_tf=True, encoding='utf-8',
                                  decode_error='ignore',
                                  max_df=0.5,
                                  min_df=10,
                                  stop_words=stopWords)

vectorised = tfidfVectorizer.fit_transform(xTrain)
print('No of Samples , No. of Features ', vectorised.shape)
''' Classifier '''

clf1 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', MultinomialNB(alpha=0.01, fit_prior=True))
])

# Best Chi square
clf2 = Pipeline([
    ('vect', tfidfVectorizer),
    ('chi2', SelectKBest(chi2, k=15000)),
    ('clf', SVC(kernel='linear'))
])

# Bernoulli Naive Bayes
clf3 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', BernoulliNB(alpha=0.01))
])

# SVC Linear Kernel
clf4 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='linear'))
])
# SVC RBF Kernel
clf5 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='rbf'))
])
# SVC Poly Kernel
clf6 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='poly'))
])



def trainAndEvaluate(clf, xTrain, xTest, yTrain, yTest):
    clf.fit(xTrain, yTrain)
    print("Accuracy on training Set : ")
    print(clf.score(xTrain, yTrain))
    print("Accuracy on Testing Set : ")
    print(clf.score(xTest, yTest))
    yPred = clf.predict(xTest)
    ''' --- START TEMPORARY ---'''
    print(str(xTest[0], encoding='utf-8'))
    print('Predicted Target ', clf.predict([xTest[0]])[0])
    print('Actual Target ', yTest[0])
    print('Predicted Target Name ', trainNews.target_names[clf.predict([xTest[0]])[0]])
    print('Actual Target Name ', trainNews.target_names[yTest[0]])

    print(str(xTest[600], encoding='utf-8'))
    print('Predicted Target ', clf.predict([xTest[600]])[0])
    print('Actual Target ', yTest[600])
    print('Predicted Target Name ', trainNews.target_names[clf.predict([xTest[600]])[0]])
    print('Actual Target Name ', trainNews.target_names[yTest[600]])

    print(str(xTest[1100], encoding='utf-8'))
    print('Predicted Target ', clf.predict([xTest[1100]])[0])
    print('Actual Target ', yTest[1100])
    print('Predicted Target Name ', trainNews.target_names[clf.predict([xTest[1100]])[0]])
    print('Actual Target Name ', trainNews.target_names[yTest[1100]])
    ''' --- END TEMPORARY ---'''
    print("Classification Report : ")
    print(metrics.classification_report(yTest, yPred))
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(yTest, yPred))


print('Multinominal Naive Bayes \n')
trainAndEvaluate(clf1, xTrain, xTest, yTrain, yTest)
print('Bernoulli Naive Bayes \n')
trainAndEvaluate(clf3, xTrain, xTest, yTrain, yTest)
print('Linear Kernel SVC \n')
trainAndEvaluate(clf4, xTrain, xTest, yTrain, yTest)
print('RBF Kernel SVC \n')
trainAndEvaluate(clf5, xTrain, xTest, yTrain, yTest)
print('Poly Kernel SVC \n')
trainAndEvaluate(clf6, xTrain, xTest, yTrain, yTest)
print('SVC With Chi Square\n')
trainAndEvaluate(clf2, xTrain, xTest, yTrain, yTest)


# Most Important Features

def showTopFeatures(classifier, vectorizer, categories, number = 25):
    featureNames = sp.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topFeatures = sp.argsort(classifier.named_steps['clf'].coef_[i])[-number:]

        print('%s: %s' %(category, " ".join(featureNames[topFeatures])))


print('Multinomial Naive Bayes \n')
showTopFeatures(clf1, tfidfVectorizer, trainNews.target_names)
print('Bernoulli Naive Bayes \n')
showTopFeatures(clf3, tfidfVectorizer, trainNews.target_names)


