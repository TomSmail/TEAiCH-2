# Import the External Libraries
from mistralai import Mistral
import datetime
from pdf2image import convert_from_path
import os
import keyboard
from dotenv import load_dotenv
import json

# Import the encode_image function from the utils.py file
from utils import encode_image

# Load environment variables from .env file
load_dotenv()

class Slides:
    def __init__(self, presentation_path):
        
        self.presentation = convert_from_path(presentation_path)
        self.current_slide = 0
        self.slide_history = [(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), 0)]

        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = Mistral(api_key=api_key)
        self.model = "pixtral-12b-2409"
        self.log_file = open('./logs/slide_log_file.txt', 'w')

    # Convert the slides to images and store them in the slides folder
    def convert_slide_to_images(self):
        for i, slide in enumerate(self.presentation):
            slide.save(f"./slides/{i}.jpg", "JPEG")

    # Detect the key press event and update the current slide
    def on_key_event(self, event):
        if event.name == "left":
            if self.current_slide > 0:
                self.current_slide -= 1
        elif event.name == "right":
            if self.current_slide < len(self.presentation) - 1:
                self.current_slide += 1
        print(f"Current slide: {self.current_slide + 1} ")
        # Export the current slide as an image
        self.slide_history.append((datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), self.current_slide))

    # Start listening for key presses
    def start(self):
        print("Listening for key presses...")
        keyboard.on_press_key("left", self.on_key_event)
        keyboard.on_press_key("right", self.on_key_event)
        # Keep the program running to listen for key presses
        keyboard.wait("esc")  # Press 'esc' to exit the program
        return self.slide_history
    
    def process_slide_history(self):
        for time, slide_num in self.slide_history:
            print(time, slide_num)
        
    def get_slide_description(self, slide_number):
        # Read the image
        image_path = f"./slides/{slide_number}.jpg"
        image_data = encode_image(image_path)
        # Send the request to the API using Mistral AI client
        messages = [{
        "role": "user",
        "content": 
            [
                {
                    "type": "text",
                    "text": 
                    """
                        Please provide a description of the slide, tell me what is being discussed in the slide.
                        Eg. For a slide depicting AI in healthcare -> "Importance of AI in healthcare."
                        I want the summary to be brief and to not use line breaks, bullet points or lists. 

                    """
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_data}" 
                }
            ]
        }]
        chat_response = self.client.chat.complete(
                model = self.model,
                messages = messages,
                response_format = {
                    "type": "json_object",
                }
            )
        print(f"Message: {chat_response.choices[0].message.content}")
        return chat_response.choices[0].message.content
    
    def log_presentation_description(self):
        descriptions = []
        for (time, slide_num)  in self.slide_history:
            try: 
                description = self.get_slide_description(slide_num)
                print(description)
                # Log the predictions
                log_entry = {
                    "slide_number": slide_num,
                    "time": time,
                    "description": description
                }
            except Exception as e:
                print(f"Error: {e}")
                log_entry = {
                    "slide_number": slide_num,
                    "time": time,
                    "description": ""
                }
            descriptions.append(log_entry)
        self.log_file.write(json.dumps(descriptions, indent=4))
        self.log_file.flush()

if __name__ == "__main__":
    our_slides = Slides("presentation.pdf")
    our_slides.convert_slide_to_images()
    print(our_slides.start())
    our_slides.log_presentation_description()