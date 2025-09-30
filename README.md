# tweet-sentiment-classifier
# Tweet Sentiment Classifier

A simple machine learning project that classifies tweets as **positive** or **negative** using NLP techniques and a Logistic Regression model.

This project demonstrates how to clean and preprocess tweet text, convert it to numerical features using TF-IDF, and train a sentiment analysis model using Scikit-learn.

---

## 🚀 Features

- Preprocessing of raw tweets (stopword removal, punctuation cleanup, etc.)
- TF-IDF vectorization for feature extraction
- Logistic Regression model for sentiment classification
- Evaluation using classification report
- Sentiment prediction for new/unseen tweets

---

## 🧰 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- NLTK (for stopwords)
- Regular Expressions (regex)

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/tweet-sentiment-classifier.git
cd tweet-sentiment-classifier
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
pip install pandas numpy scikit-learn nltk
import nltk
nltk.download('stopwords')
🧪 Usage

Run the main script:

python sentiment_classifier.py


It will:

Load and clean a sample tweet dataset

Train a sentiment classifier

Evaluate the model

Predict sentiment of new example tweets
