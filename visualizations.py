import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_attention_levels(emotion_df):
    fig = px.line(emotion_df, x='timestamp', y='attention', title='Attention Levels Over Time')
    fig.update_layout(xaxis_title='Time', yaxis_title='Attention Level')
    return fig

def plot_slide_effectiveness(emotion_df, slides_df):
    # Merge emotion and slides data
    merged_df = pd.merge_asof(emotion_df, slides_df[['timestamp', 'slide_number']], on='timestamp', direction='backward')
    
    # Calculate average attention per slide
    slide_effectiveness = merged_df.groupby('slide_number')['attention'].mean().reset_index()
    
    fig = px.bar(slide_effectiveness, x='slide_number', y='attention', title='Slide Effectiveness')
    fig.update_layout(xaxis_title='Slide Number', yaxis_title='Average Attention Level')
    return fig

def plot_emotion_distribution(emotion_df):
    emotions = ['happy', 'sad', 'angry', 'surprised', 'confused']  # Updated emotion list
    emotion_data = emotion_df[emotions].mean()
    
    fig = px.pie(values=emotion_data.values, names=emotion_data.index, title='Emotion Distribution')
    return fig

def plot_speech_sentiment(speech_df):
    fig = px.scatter(speech_df, x='start_time', y='sentiment', title='Speech Sentiment Over Time')
    fig.update_layout(xaxis_title='Time', yaxis_title='Sentiment (Positive / Negative)')
    return fig

def plot_attention_heatmap(emotion_df, slides_df):
    # Merge emotion and slides data
    merged_df = pd.merge_asof(emotion_df, slides_df[['timestamp', 'slide_number']], on='timestamp', direction='backward')
    
    # Calculate seconds since the start of the presentation
    merged_df['seconds'] = (merged_df['timestamp'] - merged_df['timestamp'].min()).dt.total_seconds()
    
    # Create a pivot table for the heatmap
    heatmap_data = merged_df.pivot_table(values='attention', index='slide_number', columns=pd.cut(merged_df['seconds'], bins=10), aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values,
                                    x=heatmap_data.columns.astype(str),
                                    y=heatmap_data.index,
                                    colorscale='RdYlGn'))
    
    fig.update_layout(title='Attention Heatmap: Slides vs Time',
                      xaxis_title='Time Segments',
                      yaxis_title='Slide Number')
    
    return fig
