# views.py

from django.shortcuts import render
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize sentiment analyzers
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_vader_sentiment(text):
    """Get VADER sentiment scores for the input text."""
    return sia.polarity_scores(text)

def get_roberta_sentiment(text):
    """Get RoBERTa sentiment scores for the input text."""
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }

def sentiment_analysis(request):
    sentiment = ''
    vader_result = {}
    roberta_result = {}
    
    if request.method == 'POST':
        text = request.POST.get('text', '').strip()
        
        # Validate input
        if not text:
            return render(request, 'sentiment/sentiment_form.html', {
                'error': 'Please enter some text.',
                'sentiment': sentiment,
                'vader_result': vader_result,
                'roberta_result': roberta_result
            })

        # Get sentiment analysis results
        try:
            vader_result = get_vader_sentiment(text)
            roberta_result = get_roberta_sentiment(text)

            # Determine sentiment from VADER
            if vader_result['compound'] > 0:
                sentiment = 'Positive'
            elif vader_result['compound'] == 0:
                sentiment = 'Neutral'
            else:
                sentiment = 'Negative'
        except Exception as e:
            return render(request, 'sentiment/sentiment_form.html', {
                'error': f'An error occurred: {str(e)}',
                'sentiment': sentiment,
                'vader_result': vader_result,
                'roberta_result': roberta_result
            })

    return render(request, 'sentiment/sentiment_form.html', {
        'sentiment': sentiment,
        'vader_result': vader_result,
        'roberta_result': roberta_result,
        'error': ''
    })
