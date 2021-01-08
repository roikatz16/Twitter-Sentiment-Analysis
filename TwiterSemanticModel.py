import os
import pandas as pd
import numpy as np
import re
import string
from scipy.sparse import csr_matrix, hstack
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import nltk

nltk.download('stopwords')
nltk.download('punkt')


class TwitterSemanticModel():
    def load_dataset(self, filename):
        dataset = pd.read_csv(filename, encoding='latin-1')
        # dataset.columns = cols
        return dataset


    #extract hashtaged words from tweet
    def get_hashtags(self, tweet):
        hashtag = re.findall(r'\#\w+', tweet)
        if len(hashtag) == 0:
            return 0
        return 1
    #extract taged words from tweet
    def get_tags(self, tweet):
        tag = re.findall(r'\@\w+', tweet)
        if len(tag) == 0:
            return 0
        return 1

    def get_tweet_length(self, tweet):
        return len(tweet)

    def count_question_marks(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        return tweet.count('?')

    def count_exclamation_marks(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        return tweet.count('!')

    def count_upper(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        return sum(map(str.isupper, tweet.split()))

    def sad_smilies(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        sad_faces = '>:[ :-( :( :-c :c :-< :< :-[ :[ :{ :| |: :/ /: 8( 8-( ]:< )-: ): >-: >: ]-: ]: }: )8 )-8'.split(' ')
        for face in sad_faces:
            if face in tweet:
                return 1
        return 0


    def happy_smilies(self, tweet):
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        happy_faces = ':-) :) :D :o) :] :3 :c) :> =] 8) =) :} :^) :-D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D (-: (: (' \
                      'o: [: <: [= (8 (= {: (^:'.split(' ')
        for face in happy_faces:
            if face in tweet:
                return 1
        return 0


    def preprocess_tweet_text(self, tweet):
        stop_words = set(stopwords.words('english'))
        tweet.lower()
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in stop_words]

        # ps = PorterStemmer()
        # stemmed_words = [ps.stem(w) for w in filtered_words]
        # lemmatizer = WordNetLemmatizer()
        # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

        return " ".join(filtered_words)

    def get_feature_vector(self, train_fit):
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector

    def amount_positive_negative(self, tweets):
        # Global Parameters
        stop_words = set(stopwords.words('english'))
        positive_words_txt = os.path.join('positive_words.txt')
        negative_words_txt = os.path.join('negative_words.txt')

        pos_words = []
        neg_words = []

        for pos_word in open(positive_words_txt, 'r').readlines():
            # pos_words.append(({pos_word.rstrip(): True}, 'positive'))
            pos_words.append(pos_word.rstrip())

        for neg_word in open(negative_words_txt, 'r').readlines():
            # neg_words.append(({neg_word.rstrip(): True}, 'negative'))
            neg_words.append(neg_word.rstrip())

        a = []
        positives = []
        negatives = []
        for tweet in tweets:
            words = tweet.split()
            positive_counter, negative_counter = 0, 0
            for word in words:
                if word in pos_words:
                    positive_counter += 1
                if word in neg_words:
                    negative_counter += 1
            positives.append(positive_counter)
            negatives.append(negative_counter)
        a.append(positives)
        a.append(negatives)
        return csr_matrix(a).transpose()

    def int_to_string(self, sentiment):
        if sentiment == 0:
            return "Negative"
        elif sentiment == 2:
            return "Neutral"
        else:
            return "Positive"

    def get_tweet_features_by_TfidfVectorizer(self, dataset, tf_vector):
        X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
        return X

    def cross_validation(self, X_train, Y_train, scoring):
        models = []
        models.append(("LR", LogisticRegression(solver='lbfgs')))
        models.append(("KNN", KNeighborsClassifier()))
        # models.append(("NB", GaussianNB()))
        models.append(("NB", MultinomialNB()))
        models.append(("SVM", svm.SVC()))
        results = []
        names = []
        for name, model in models:
            cv_results = cross_val_score(model, X_train, Y_train, cv=10, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = f"{name} {scoring}, mean:{cv_results.mean()}, std:{cv_results.std()}"
            print(msg)
        return results, models

    def fit_predict_model(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        return y_predict

    def combine_features(self, X, other_features):
        # other_feature should be a list
        csr_features = csr_matrix(other_features).transpose()
        combined_features = hstack((X, csr_features))
        return combined_features



if __name__ == '__main__':
    TSM = TwitterSemanticModel()
    # Load dataset
    dataset = TSM.load_dataset("Train.csv")
    # dataset = dataset.iloc[:200]

    # Preprocess data
    hashtags = (dataset['SentimentText'].apply(TSM.get_hashtags)).tolist()
    tags = (dataset['SentimentText'].apply(TSM.get_tags)).tolist()
    tweet_length = (dataset['SentimentText'].apply(TSM.get_tweet_length)).tolist()

    feature_question_marks = dataset['SentimentText'].apply(TSM.count_question_marks).to_list()
    feature_exclamation_marks = dataset['SentimentText'].apply(TSM.count_exclamation_marks).to_list()
    feature_upper_words = dataset['SentimentText'].apply(TSM.count_upper).to_list()

    feature_sad = dataset['SentimentText'].apply(TSM.sad_smilies).to_list()
    feature_happy = dataset['SentimentText'].apply(TSM.happy_smilies).to_list()

    dataset.SentimentText = dataset['SentimentText'].apply(TSM.preprocess_tweet_text)
    # Same tf vector will be used for Testing sentiments on unseen trending data
    tf_vector = TSM.get_feature_vector(np.array(dataset.iloc[:, 1]).ravel())
    X = TSM.get_tweet_features_by_TfidfVectorizer(dataset, tf_vector)

    other_features = []
    # TODO: add to combine_features more features
    # other_features.append(hashtags)
    # other_features.append(tags)
    # other_features.append(tweet_length)

    # other_features.append(feature_question_marks)
    # other_features.append(feature_exclamation_marks)
    # other_features.append(feature_upper_words)

    other_features.append(feature_sad)
    other_features.append(feature_happy)

    X = TSM.combine_features(X, other_features)

    y = np.array(dataset.iloc[:, 0]).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    """ ----------- CROSS VALIDATION ----------- """
    cross_val_results, models = TSM.cross_validation(X_train, y_train, scoring='accuracy')
    cross_val_results, models = TSM.cross_validation(X_train, y_train, scoring='precision')
    cross_val_results, models = TSM.cross_validation(X_train, y_train, scoring='recall')
    """-------------TRAINING MODEL------------"""
    for name, model in models:
        # model = MultinomialNB()  # choose model
        y_predict = TSM.fit_predict_model(model, X_train, y_train, X_test)
        """-------------GET ACCURACY SCORE------------"""
        msg = f"{name}: {accuracy_score(y_test, y_predict)}"
        print(msg)
        # print(accuracy_score(y_test, y_predict))

        """-------GET TEST PREDICT RESULTS FOR KAGGEL CHALLANGE------"""
        # Load dataset
        test_file_name = "Test.csv"
        test_ds = TSM.load_dataset(test_file_name)
        # Preprocess data
        hashtags = (test_ds['SentimentText'].apply(TSM.get_hashtags)).tolist()
        tags = (test_ds['SentimentText'].apply(TSM.get_tags)).tolist()
        tweet_length = (test_ds['SentimentText'].apply(TSM.get_tweet_length)).tolist()

        feature_question_marks = test_ds['SentimentText'].apply(TSM.count_question_marks).to_list()
        feature_exclamation_marks = test_ds['SentimentText'].apply(TSM.count_exclamation_marks).to_list()
        feature_upper_words = test_ds['SentimentText'].apply(TSM.count_upper).to_list()

        feature_sad = test_ds['SentimentText'].apply(TSM.sad_smilies).to_list()
        feature_happy = test_ds['SentimentText'].apply(TSM.happy_smilies).to_list()

        test_ds.SentimentText = test_ds['SentimentText'].apply(TSM.preprocess_tweet_text)

        X = TSM.get_tweet_features_by_TfidfVectorizer(test_ds, tf_vector)
        other_features = []
        """--------PUT FEATURES---------------"""
        # other_features.append(hashtags)
        # other_features.append(tags)
        # other_features.append(tweet_length)

        # other_features.append(feature_question_marks)
        # other_features.append(feature_exclamation_marks)
        # other_features.append(feature_upper_words)

        other_features.append(feature_sad)
        other_features.append(feature_happy)

        X = TSM.combine_features(X, other_features)

        y_predict = model.predict(X)
        # save results
        test_result_ds = pd.DataFrame({'ID': test_ds["ID"], 'Sentiment': y_predict})
        test_result_ds.to_csv(f"real_sample_'{name}'.csv'", index=False)
