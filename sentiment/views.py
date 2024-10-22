from django.shortcuts import render
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline  # Updated imports
from scipy.special import softmax
from googletrans import Translator  # You may need to install this library

# Initialize VADER sentiment analyzer for English
sia = SentimentIntensityAnalyzer()

# Load multilingual RoBERTa for multilingual sentiment analysis
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  # This can be replaced with a multilingual model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
translator = Translator()

# Initialize the Hugging Face sentiment-analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

def get_vader_sentiment(text):
    """Get VADER sentiment scores for English text."""
    return sia.polarity_scores(text)

def get_roberta_sentiment(text):
    """Get RoBERTa sentiment scores for multilingual text."""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }

def detect_language(text):
    """Detect if text is English or other languages. Return True if English."""
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def sentiment_analysis(request):
    sentiment = ''
    vader_result = {}
    roberta_result = {}
    pipeline_result = {}
    
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        
        # Validate input
        if not text:
            return render(request, 'sentiment/sentiment_form.html', {
                'error': 'Please enter some text.',
                'sentiment': sentiment,
                'vader_result': vader_result,
                'roberta_result': roberta_result,
                'pipeline_result': pipeline_result
            })

        try:
            # Detect if the text is in English or not
            is_english = detect_language(text)

            # Translate to English if the text is not in English
            if not is_english:
                translated_text = translator.translate(text, src='auto', dest='en').text
            else:
                translated_text = text

            # Get sentiment analysis using VADER (English)
            vader_result = get_vader_sentiment(translated_text)

            # Get sentiment analysis using RoBERTa (Multilingual)
            roberta_result = get_roberta_sentiment(translated_text)

            # Get sentiment analysis using the Hugging Face pipeline
            pipeline_result = sent_pipeline(translated_text)

            # Determine sentiment from the results
            if vader_result.get('compound', 0) > 0 or roberta_result['roberta_pos'] > max(roberta_result['roberta_neg'], roberta_result['roberta_neu']):
                sentiment = 'Positive'
            elif vader_result.get('compound', 0) == 0 or roberta_result['roberta_neu'] > max(roberta_result['roberta_neg'], roberta_result['roberta_pos']):
                sentiment = 'Neutral'
            else:
                sentiment = 'Negative'
                
        except Exception as e:
            return render(request, 'sentiment/sentiment_form.html', {
                'error': f'An error occurred: {str(e)}',
                'sentiment': sentiment,
                'vader_result': vader_result,
                'roberta_result': roberta_result,
                'pipeline_result': pipeline_result
            })

    return render(request, 'sentiment/sentiment_form.html', {
        'sentiment': sentiment,
        'vader_result': vader_result,
        'roberta_result': roberta_result,
        'pipeline_result': pipeline_result,
        'error': ''
    })
