import pandas as pd
# Making DataFrame 
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' , quoting = 3)

#cleaning Text 
import re
import nltk
nltk.download('stopwords')  # Stopwords used to find the words that are not useful in Review Analysis.

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


corpus = [] # Used for storing the important root words from the reviews
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])   # Getting valid strings from the reviews.
    review = review.lower()                            # Getting review in lower case
    review = review.split()                            # Splitting review into seperate word's list
    ps = PorterStemmer()                               # Creating PortStemmer Object
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #Getting root words of the important words from review.
    review = ' '.join(review)                          # Joining the root words into a string 
    corpus.append(review)                              # Appending the string to corpus

# create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
    
cv = CountVectorizer(max_features = 600) # Creates 600 word vector which are repeating in the reviews.
X = cv.fit_transform(corpus).toarray()   # Cretes a Matrix to fit 1000 reviews into vector.

y = df.iloc[:,-1].values                # Geting 'likes' colunm to y variable 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)  # Last 200 entries are for testing and 800 entries for training the Model.


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train) # Fitting the Training and Testing Data to Classifier



# Predicting the Test set results
y_pred = classifier.predict(X_test) # Predicting the values based on X-test Data.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)













