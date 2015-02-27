import nltk
import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    print "Loading train set..."
    train_set = pandas.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

    print "Building vertorizer and getting features for train set..."
    vectorizer = TfidfVectorizer(
        analyzer='word',
        strip_accents='unicode',
        tokenizer=nltk.word_tokenize,
        stop_words=nltk.corpus.stopwords.words('english'),
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        max_features=5000
    )
    train_set_features = vectorizer.fit_transform(train_set['review'], train_set['sentiment']).toarray()

    print "Training random forest..."
    forest = RandomForestClassifier(n_estimators=100, n_jobs=4)
    forest = forest.fit(train_set_features, train_set['sentiment'])

    print "Clean up train data..."
    del train_set
    del train_set_features

    print "Loading test set..."
    test_set = pandas.read_csv('data/testData.tsv', header=0, delimiter='\t', quoting=3)

    print "Getting features for test set... "
    test_set_features = vectorizer.transform(test_set['review']).toarray()

    print "Get predictions..."
    predictions = forest.predict_proba(test_set_features)

    print "Wrting predictions to disk..."
    output = pandas.DataFrame(data={'id': test_set['id'], 'sentiment': predictions[:,0]})
    output.to_csv('data/bag_of_words_2gram_tfidf.csv', index=False, quoting=3)

    print "Done"
