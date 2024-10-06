import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import load_and_process_data
from visualizations import plot_attention_levels, plot_slide_effectiveness, plot_emotion_distribution, plot_speech_sentiment, plot_attention_heatmap
from analysis import generate_recommendations, analyze_keywords
from chatbot import get_chatbot_response
import time
import threading

# Load and process data
emotion_df, speech_df, slides_df = load_and_process_data()

st.set_page_config(page_title="Presentation Analytics", layout="wide")

st.title("Presentation Analytics Dashboard")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["Overview", "Attention Analysis", "Slide Effectiveness", "Chatbot"])

# Initialize session state
if 'script_running' not in st.session_state:
    st.session_state.script_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0


def run_script():
    st.session_state.script_running = True
    st.session_state.start_time = time.time()
    while st.session_state.script_running:
        time.sleep(1)
        st.session_state.elapsed_time = int(time.time() -
                                            st.session_state.start_time)


def stop_script():
    st.session_state.script_running = False


if page == "Overview":
    st.header("Overview")
    st.write(
        "Welcome to the Presentation Analytics Dashboard. Use the sidebar to navigate between different metrics and analyses."
    )

    # Display basic stats
    st.subheader("Presentation Summary")
    duration = speech_df['end_time'].max() - speech_df['start_time'].min()
    st.write(f"Total duration: {duration.total_seconds():.2f} seconds")
    st.write(f"Number of slides: {slides_df['slide_number'].max()}")
    st.write(f"Average attention level: {emotion_df['attention'].mean():.2f}")

    # Emotion Distribution
    st.subheader("Emotion Distribution")
    fig_emotion_dist = plot_emotion_distribution(emotion_df)
    st.plotly_chart(fig_emotion_dist, use_container_width=True, height=600)

    # Keyword Analysis
    st.subheader("Top Keywords")
    keyword_counts = analyze_keywords(speech_df)
    fig_keywords = px.bar(keyword_counts.head(10),
                          x=keyword_counts.head(10).index,
                          y=keyword_counts.head(10).values)
    fig_keywords.update_layout(xaxis_title="Keyword", yaxis_title="Frequency")
    st.plotly_chart(fig_keywords, use_container_width=True)

elif page == "Attention Analysis":
    st.header("Attention Level Analysis")

    # Plot attention levels over time
    fig_attention = plot_attention_levels(emotion_df)
    st.plotly_chart(fig_attention, use_container_width=True)

    # Attention Heatmap
    st.subheader("Attention Heatmap")
    fig_heatmap = plot_attention_heatmap(emotion_df, slides_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Identify attention dips
    attention_threshold = st.slider("Attention dip threshold", 0.0, 1.0, 0.5)
    attention_dips = emotion_df[emotion_df['attention'] < attention_threshold]

    st.subheader("Moments of Low Attention")
    for _, dip in attention_dips.iterrows():
        st.write(
            f"Time: {(dip['timestamp'] - emotion_df['timestamp'].min()).total_seconds():.2f}s, Attention: {dip['attention']:.2f}"
        )

elif page == "Slide Effectiveness":
    st.header("Slide Effectiveness")

    # Plot slide effectiveness
    fig_slides = plot_slide_effectiveness(emotion_df, slides_df)
    st.plotly_chart(fig_slides, use_container_width=True)

    # Display slide-specific insights
    st.subheader("Slide Insights")
    for _, slide in slides_df.iterrows():
        st.write(
            f"Slide {slide['slide_number']}: Avg. Attention: {slide['avg_attention']:.2f}"
        )
        st.write(f"Content: {slide['content_summary']}")

elif page == "Speech Analysis":
    st.header("Speech Sentiment Analysis")

    # Plot speech sentiment
    fig_sentiment = plot_speech_sentiment(speech_df)
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Display speech insights
    st.subheader("Speech Insights")
    avg_sentiment = speech_df['sentiment'].mean()
    st.write(f"Average sentiment: {avg_sentiment:.2f}")
    if avg_sentiment > 0:
        st.write("Overall positive sentiment in the speech.")
    elif avg_sentiment < 0:
        st.write("Overall negative sentiment in the speech.")
    else:
        st.write("Neutral sentiment in the speech.")

    # Display speech segments
    st.subheader("Speech Segments")
    for _, segment in speech_df.iterrows():
        st.write(
            f"Time: {segment['start_time'].strftime('%M:%S')} - {segment['end_time'].strftime('%M:%S')}"
        )
        st.write(f"Text: {segment['text']}")
        st.write(f"Keywords: {segment['keywords']}")
        st.write(f"Sentiment: {segment['sentiment']}")
        st.write("---")

elif page == "Recommendations":
    st.header("Engagement Recommendations")

    recommendations = generate_recommendations(emotion_df, speech_df,
                                               slides_df)
    for category, rec_list in recommendations.items():
        st.subheader(category)
        for rec in rec_list:
            st.write(f"- {rec}")

elif page == "Chatbot":
    st.header("Presentation Analytics Chatbot")
    st.write(
        "Ask questions about your presentation analytics, and I'll do my best to help!"
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input(
            "What would you like to know about your presentation?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = get_chatbot_response(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

elif page == "Script Management":
    st.header("Start The App")

    col1, col2 = st.columns(2)

    with col1:
        if not st.session_state.script_running:
            if st.button("Run"):
                threading.Thread(target=run_script).start()
        else:
            if st.button("Stop"):
                stop_script()

    with col2:
        st.write(f"Elapsed Time: {st.session_state.elapsed_time} seconds")

    if st.session_state.script_running:
        st.write("Script is running...")
    else:
        st.write("Script is stopped.")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Presentation Analytics Dashboard v1.0")
