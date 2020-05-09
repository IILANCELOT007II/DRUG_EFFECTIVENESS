import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, 1:6].values
Y_train = dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1= LabelEncoder()
X_train[:, 0] = labelencoder_X_1.fit_transform(X_train[:, 0])
labelencoder_X_2= LabelEncoder()
X_train[:, 1] = labelencoder_X_2.fit_transform(X_train[:, 1])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 32165):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review_by_patient'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


X_train[:,2] = corpus
X1_train = pd.DataFrame(X_train)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
X1_train['VaderScore'] = X1_train[2].apply(lambda x: analyzer.polarity_scores(x)['compound'])

X1_train['VaderScore'] = X1_train['VaderScore'].map(lambda x: int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0))
X1_train['ratingScore'] = X1_train[3].map(lambda x: int(2) if x>=7 else int(1) if x<=3 else int(0))
X1_train = X1_train.drop([2], axis=1)

df = pd.read_csv('test1.csv')
X_test = df.iloc[:, 1:6].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_3= LabelEncoder()
X_test[:, 0] = labelencoder_X_3.fit_transform(X_test[:, 0])
labelencoder_X_4= LabelEncoder()
X_test[:, 1] = labelencoder_X_4.fit_transform(X_test[:, 1])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 10760):
    review = re.sub('[^a-zA-Z]', ' ', df['review_by_patient'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


X_test[:,2] = corpus
X1_test = pd.DataFrame(X_test)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
X1_test['VaderScore'] = X1_test[2].apply(lambda x: analyzer.polarity_scores(x)['compound'])

X1_test['VaderScore'] = X1_test['VaderScore'].map(lambda x: int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0))
X1_test['ratingScore'] = X1_test[3].map(lambda x: int(2) if x>=7 else int(1) if x<=3 else int(0))
X1_test = X1_test.drop([2], axis=1)




from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)



from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
model = XGBRegressor()
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid = GridSearchCV(model,
                        parameters,
                        cv = 10,
                        n_jobs = -1,
                        verbose=True)
grid.fit(X1_train, Y_train)
y_pred = grid.predict(X1_test)
