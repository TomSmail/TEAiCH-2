import pandas as pd
import json
from textblob import TextBlob
import os


def load_json(file_path):
    full_path = os.path.join(os.getcwd(), file_path)
    with open(full_path, 'r') as f:
        return json.load(f)


def process_emotion_data(data):
    processed_data = []
    for entry in data:
        timestamp = pd.to_datetime(entry['timestamp'], format='%Y%m%d_%H%M%S')
        listeners = entry['listeners']

        emotion_levels = {
            'happiness': 0,
            'sadness': 0,
            'anger': 0,
            'surprise': 0,
            'confusion': 0
        }
        attention_sum = 0

        for listener in listeners:
            listener_data = listener['data']
            for emotion, level in emotion_levels.items():
                emotion_levels[emotion] += listener_data[f'{emotion}_level']
            attention_sum += listener_data['attention_level']

        total_listeners = len(listeners)
        processed_data.append({
            'timestamp': timestamp,
            'happy': emotion_levels['happiness'] / total_listeners,
            'sad': emotion_levels['sadness'] / total_listeners,
            'angry': emotion_levels['anger'] / total_listeners,
            'surprised': emotion_levels['surprise'] / total_listeners,
            'confused': emotion_levels['confusion'] / total_listeners,
            'attention': attention_sum / total_listeners
        })

    return pd.DataFrame(processed_data)


def process_speech_data(data):
    processed_data = []
    for entry in data:
        start_time = pd.to_datetime(entry['start_time'],
                                    format='%Y%m%d_%H%M%S')
        end_time = pd.to_datetime(entry['end_time'], format='%Y%m%d_%H%M%S')

        processed_data.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': entry['text'],
            'keywords': ', '.join(entry['keywords']),
            'sentiment': entry['sentiment']
        })

    return pd.DataFrame(processed_data)


def process_slides_data(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    df['slide_number'] = range(1, len(df) + 1)
    return df


def synchronize_data(emotion_df, speech_df, slides_df):
    # Merge dataframes based on timestamp
    merged_df = pd.merge_asof(emotion_df,
                              speech_df,
                              left_on='timestamp',
                              right_on='start_time',
                              direction='forward')
    merged_df = pd.merge_asof(merged_df,
                              slides_df,
                              on='timestamp',
                              direction='backward')

    # Calculate average attention per slide
    slides_df['avg_attention'] = merged_df.groupby(
        'slide_number')['attention'].mean()

    return emotion_df, speech_df, slides_df


def load_and_process_data():
    emotion_data = load_json('data/emotion_log.json')
    speech_data = load_json('data/speech_log.json')
    slides_data = load_json('data/slides_log.json')

    emotion_df = process_emotion_data(emotion_data)
    speech_df = process_speech_data(speech_data)
    slides_df = process_slides_data(slides_data)

    return synchronize_data(emotion_df, speech_df, slides_df)

# Test function to check if load_json works correctly
def test_load_json():
    try:
        slides_data = load_json('data/slides_log.json')
        print("Successfully loaded slides_log.json")
        print(f"Number of slides: {len(slides_data)}")
        print("First slide data:", slides_data[0])
        return True
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Run the test function
if __name__ == "__main__":
    test_result = test_load_json()
    print(f"Test result: {'Passed' if test_result else 'Failed'}")
