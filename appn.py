from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import joblib
from pygooglenews import GoogleNews
from bs4 import BeautifulSoup
from translate import Translator
from textblob import TextBlob
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)

# Load the RandomForest model and the vectorizer using joblib
rf_model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Function to clean HTML content
def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# Function to classify news text
def classify_text(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    
    # Use RandomForest model for prediction
    rf_pred = rf_model.predict(input_vector)
    
    class_names = ['world', 'sports', 'business', 'science']
    rf_result = class_names[rf_pred[0]]
    
    return rf_result

# Function to fetch news titles from Google News
def get_titles(keyword, start_date=None, end_date=None):
    news = []
    gn = GoogleNews(lang='en', country='IN')
    search = gn.search(keyword)
    articles = search['entries']
    
    # Convert input dates to datetime.date for comparison
    if start_date:
        start_date = pd.to_datetime(start_date).date()
    if end_date:
        end_date = pd.to_datetime(end_date).date()
    
    for i in articles:
        article_date = pd.to_datetime(i.published).date()
        
        # Filter articles based on the date range
        if (start_date and end_date):
            if start_date <= article_date <= end_date:
                article = {'title': i.title, 'link': i.link, 'published': i.published}
                news.append(article)
        else:
            article = {'title': i.title, 'link': i.link, 'published': i.published}
            news.append(article)
    
    return news

# Function to translate text using translate library
def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    try:
        translated = translator.translate(text)
    except Exception as e:
        translated = text
    return translated

# Function to get sentiment
def sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Route to generate news report
@app.route('/generate_report', methods=['GET', 'POST'])
def generate_report():
    if request.method == 'POST':
        keyword = request.form['keyword']
        target_language = request.form['language']
        period = request.form['period']
        custom_period = request.form.get('custom_period')
        
        # Determine date range based on period
        if period == 'day':
            start_date = pd.Timestamp.now() - pd.DateOffset(days=1)
            end_date = pd.Timestamp.now()
        elif period == 'week':
            start_date = pd.Timestamp.now() - pd.DateOffset(weeks=1)
            end_date = pd.Timestamp.now()
        elif period == 'month':
            start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
            end_date = pd.Timestamp.now()
        elif period == 'year':
            start_date = pd.Timestamp.now() - pd.DateOffset(years=1)
            end_date = pd.Timestamp.now()
        elif period == 'custom':
            try:
                # Custom period should be in the format 'YYYY-MM-DD - YYYY-MM-DD'
                start_date_str, end_date_str = custom_period.split(' - ')
                start_date = pd.to_datetime(start_date_str)
                end_date = pd.to_datetime(end_date_str)
            except ValueError as e:
                # Handle invalid date format
                start_date = None
                end_date = None
        else:
            start_date = None
            end_date = None
        
        # Fetch news data
        news_data = get_titles(keyword, start_date, end_date)
        df = pd.DataFrame(news_data)
        
        if not df.empty:
            # Translate titles and perform sentiment analysis
            df['translation'] = df['title'].apply(lambda x: translate_text(x, target_language))
            df['summary'] = df['translation'].apply(lambda x: clean_html(x))
            df['Sentiment'] = df['summary'].apply(sentiment)
            df['Sentiment'] = np.where(df['Sentiment'] < 0, "Negative",
                                         np.where(df['Sentiment'] > 0, "Positive",
                                                  "Neutral"))
            
            # Process date
            df['Date'] = pd.to_datetime(df['published']).dt.date
            df = df.sort_values(by='Date', ascending=False)
        else:
            df = pd.DataFrame(columns=['title', 'translation', 'summary', 'Sentiment', 'Date'])
        
        return render_template('newdisplayreport.html', news_list=df.to_dict(orient='records'))
    
    return render_template('newgenerate_report.html')

# Route to classify user input text
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        user_input = request.form['user_input']
        result = classify_text(user_input)
        return render_template('classify.html', result=result)
    return render_template('newclassify.html')

@app.route('/')
def home():
    return render_template('newhome1.html')

if __name__ == '__main__':
    app.run(debug=True)
