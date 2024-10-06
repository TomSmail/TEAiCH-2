import pandas as pd
import numpy as np
from textblob import TextBlob

def identify_attention_dips(emotion_df, threshold=0.5):
    return emotion_df[emotion_df['attention'] < threshold]

def analyze_slide_effectiveness(slides_df):
    return slides_df.sort_values('avg_attention', ascending=False)

def analyze_speech_sentiment(speech_df):
    # Sentiment is already provided in the new format, so we don't need to calculate it
    return speech_df

def analyze_keywords(speech_df):
    all_keywords = [kw for keywords in speech_df['keywords'].str.split(', ') for kw in keywords]
    return pd.Series(all_keywords).value_counts()

def generate_recommendations(emotion_df, speech_df, slides_df):
    recommendations = {
        "Attention Improvement": [],
        "Slide Design": [],
        "Speech Delivery": [],
        "Content Optimization": []
    }

    # Attention Improvement
    attention_dips = identify_attention_dips(emotion_df)
    if not attention_dips.empty:
        dip_times = (attention_dips['timestamp'] - emotion_df['timestamp'].min()).dt.total_seconds().tolist()
        recommendations["Attention Improvement"].append(f"Consider making your content more engaging during low attention periods, especially around {', '.join([f'{t:.0f}s' for t in dip_times])}.")
        recommendations["Attention Improvement"].append("Use interactive elements or questions to re-engage the audience during attention dips.")

    # Slide Design
    slide_effectiveness = analyze_slide_effectiveness(slides_df)
    least_effective_slides = slide_effectiveness.tail(3)['slide_number'].tolist()
    most_effective_slides = slide_effectiveness.head(3)['slide_number'].tolist()
    recommendations["Slide Design"].append(f"Review and improve slides {', '.join(map(str, least_effective_slides))} for better engagement.")
    recommendations["Slide Design"].append(f"Learn from the success of slides {', '.join(map(str, most_effective_slides))} and apply similar techniques to less effective slides.")
    recommendations["Slide Design"].append("Consider using more visuals or interactive elements in less effective slides.")

    # Speech Delivery
    speech_sentiment = analyze_speech_sentiment(speech_df)
    avg_sentiment = speech_sentiment['sentiment'].mean()
    sentiment_variance = speech_sentiment['sentiment'].var()
    
    if avg_sentiment < 0:
        recommendations["Speech Delivery"].append("Try to use more positive language in your presentation to improve overall sentiment.")
    elif avg_sentiment > 0.5:
        recommendations["Speech Delivery"].append("Your positive tone is effective. Keep it up!")
    else:
        recommendations["Speech Delivery"].append("Consider varying your tone to emphasize key points and maintain audience interest.")
    
    if sentiment_variance < 0.1:
        recommendations["Speech Delivery"].append("Your speech sentiment is consistent. Consider adding more emotional variety to keep the audience engaged.")
    elif sentiment_variance > 0.5:
        recommendations["Speech Delivery"].append("Your speech has a good variety of sentiment. Make sure the changes in tone align with your content.")

    # Content Optimization
    keyword_counts = analyze_keywords(speech_df)
    top_keywords = keyword_counts.head(5).index.tolist()
    recommendations["Content Optimization"].append(f"The most frequently used keywords are: {', '.join(top_keywords)}. Ensure these align with your main message.")
    
    if len(keyword_counts) < 10:
        recommendations["Content Optimization"].append("Consider expanding your vocabulary and using a wider range of terms to keep the audience engaged.")
    
    long_segments = speech_df[speech_df['end_time'] - speech_df['start_time'] > pd.Timedelta(seconds=60)]
    if not long_segments.empty:
        recommendations["Content Optimization"].append(f"You have {len(long_segments)} speech segments longer than 60 seconds. Consider breaking these into shorter, more focused points.")

    # General recommendations
    recommendations["Attention Improvement"].append("Break up long sections with short activities or discussions.")
    recommendations["Slide Design"].append("Ensure your slides are not text-heavy. Use bullet points and images effectively.")
    recommendations["Speech Delivery"].append("Practice pacing your speech and using pauses for emphasis.")

    return recommendations
