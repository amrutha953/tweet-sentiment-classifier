
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re



nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



data = {
    'tweet': [
        'I love this new phone! So cool!',
        'Worst experience ever. Do not recommend!',
        'Just okay, not bad but not great.',
        'I am extremely happy with the service.',
        'Terrible. Waste of money.',
        'Absolutely fantastic product!',
        'Horrible, very disappointed.',
        'Best thing I’ve bought this year.',
        'It’s fine. Not the best.',
        'Completely useless.'
    ],
    'sentiment': [
        'positive',
        'negative',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'negative'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)
df.head()



def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # remove @mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

# Apply cleaning to tweets
df['clean_tweet'] = df['tweet'].apply(clean_text)
df[['tweet', 'clean_tweet']]



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LogisticRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))



new_tweets = [
    "I hate this app. It's awful!",
    "Wow, I really enjoyed using this app!",
    "Meh, nothing special.",
    "The service was fantastic and quick."
]

# Clean and vectorize
cleaned_new = [clean_text(t) for t in new_tweets]
X_new = vectorizer.transform(cleaned_new)
predictions = model.predict(X_new)

# Show predictions
for tweet, sentiment in zip(new_tweets, predictions):
    print(f"Tweet: {tweet} --> Sentiment: {sentiment}")



