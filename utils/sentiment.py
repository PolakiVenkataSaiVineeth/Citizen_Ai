from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, Any

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using both TextBlob and VADER,
    then combine the results for more accurate sentiment analysis.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with sentiment classification and score
    """
    # Get TextBlob sentiment
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    
    # Get VADER sentiment
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    # Combine scores (weighted average, giving more weight to VADER)
    combined_score = (vader_compound * 0.7) + (textblob_polarity * 0.3)
    
    # Classify sentiment
    if combined_score >= 0.05:
        sentiment = "positive"
    elif combined_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": combined_score,
        "textblob_score": textblob_polarity,
        "vader_score": vader_compound,
        "vader_details": vader_scores
    }

def extract_keywords(text: str, top_n: int = 5) -> list:
    """
    Extract important keywords from text.
    This is a simple implementation that could be enhanced with more sophisticated NLP techniques.
    
    Args:
        text: The text to analyze
        top_n: Number of top keywords to return
        
    Returns:
        List of top keywords
    """
    # Convert to lowercase and tokenize
    blob = TextBlob(text.lower())
    
    # Get word frequencies, excluding common stop words
    word_freq = {}
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "about", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they",
        "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"
    }
    
    for word in blob.words:
        if len(word) > 2 and word not in stop_words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]