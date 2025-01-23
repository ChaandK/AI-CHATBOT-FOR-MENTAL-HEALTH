# import streamlit as st
# import json
# import numpy as np
# import random
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import nltk
# from nltk.stem import WordNetLemmatizer
# from sklearn.preprocessing import LabelEncoder
# import pickle


# # Initialize NLTK components
# nltk.download('punkt')
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

# # Load the trained Keras model
# model = load_model("chatbot_model.keras")

# # Load the intents JSON file
# with open("intents.json") as file:
#     intents = json.load(file)

# # Preprocess intents data
# words = []   # Vocabulary
# classes = [] # Output labels
# documents = [] # Tuples of (patterns, tag)

# # Prepare word tokenizer and class list
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Tokenize each word
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#     if intent['tag'] not in classes:
#         classes.append(intent['tag'])

# # Lemmatize and remove duplicates
# words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!', '.', ',']]))
# classes = sorted(set(classes))



# # Load vocabulary and classes
# with open("model_words.pkl", "rb") as f:
#     words = pickle.load(f)

# with open("classes.pkl", "rb") as f:
#     classes = pickle.load(f)

# # Ensure the input preprocessing uses the exact loaded vocabulary
# def preprocess_input(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
#     bag = [1 if word in tokens else 0 for word in words]
#     return np.array([bag])

# def get_response(tag):
#     for intent in intents['intents']:
#         if intent['tag'] == tag:
#             return random.choice(intent['responses'])
#     return "I'm sorry, I don't understand."

# # Streamlit UI
# st.title("Mental Health Chatbot")
# st.markdown("""
# This AI chatbot is here to help with mental health inquiries. Please note that this is not a replacement for professional help.
# """)

# # Input field
# user_input = st.text_input("Ask me something about mental health", "")

# # Handle user input
# if user_input:
#     input_data = preprocess_input(user_input)
#     prediction = model.predict(input_data)[0]
#     confidence_index = np.argmax(prediction)
#     confidence_score = prediction[confidence_index]
#     tag = classes[confidence_index]

#     if confidence_score > 0.7: # Confidence threshold
#         response = get_response(tag)
#     else:
#         response = "I'm sorry, I didn't quite understand that. Can you rephrase it?"

#     # Display response
#     st.markdown(f"**Chatbot:** {response}")


# import os  
# import pickle  
# import json  
# import re  
# import numpy as np  
# import nltk  
# import streamlit as st  
# from nltk.stem import WordNetLemmatizer  
# from tensorflow.keras.models import load_model  

# # Load the model and necessary files  
# model = load_model('chatbot_model.keras')  
# lemmatizer = WordNetLemmatizer()  

# # Load intents and other necessary data  
# intents = json.load(open('intents.json'))  
# words = pickle.load(open('model_words.pkl', 'rb'))  
# classes = pickle.load(open('model_classes.pkl', 'rb'))  

# # Function to preprocess user input  
# def preprocess_input(user_input):  
#     # Tokenize and lemmatize input  
#     cleaned_input = re.sub(r"[^\w\s]", "", user_input)  
#     tokenized_input = nltk.word_tokenize(cleaned_input)  
#     bag = [0] * len(words)  
    
#     for word in tokenized_input:  
#         if word.lower() in words:  
#             bag[words.index(word.lower())] = 1  
#     return np.array(bag)  

# # Function to get response from model  
# def get_response(user_input):  
#     bag = preprocess_input(user_input)  
#     prediction = model.predict(np.array([bag]))[0]  
#     intent_index = np.argmax(prediction)  
#     intent_tag = classes[intent_index]  
    
#     for intent in intents['intents']:  
#         if intent['tag'] == intent_tag:  
#             response = np.random.choice(intent['responses'])  
#             return response  
#     return "I'm sorry, I didn't understand that." 
     

# # Streamlit app layout  
# st.title("Chatbot Interface")  
# st.subheader("Ask me anything about mental health!")  

# # Chat interface  
# if 'chat_history' not in st.session_state:  
#     st.session_state.chat_history = []  

# # user_input = st.text_input("You: ", "")  
# # if st.button("Send"):  
# #     if user_input:  
# #         response = get_response(user_input)  
# #         st.session_state.chat_history.append({"user": user_input, "bot": response})  
# #         st.text_input("You: ", "", key="input")  # Clear input box  

# # Display chat history  
# for chat in st.session_state.chat_history:  
#     st.write(f"You: {chat['user']}")  
#     st.write(f"Bot: {chat['bot']}")


# # Input text box for user  
# user_input = st.text_input("You: ", "")  
# if st.button("Send"):  
#     if user_input:  
#         response = get_response(user_input)  
#         st.session_state.chat_history.append({"user": user_input, "bot": response})  

# # Optional: Automatically scroll to the bottom of chat history  
# if st.session_state.chat_history:  
#     st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# import random
# import json
# import pickle
# import nltk
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')

# # Load required files
# lemmatizer = WordNetLemmatizer()

# # Load the trained Keras model
# model = load_model("chatbot_model.keras")

# # Load words and classes
# with open("model_words.pkl", "rb") as f:
#     words = pickle.load(f)

# with open("model_classes.pkl", "rb") as f:
#     classes = pickle.load(f)

# # Load intents
# with open("intents.json", "r") as f:
#     intents = json.load(f)

# # Function to preprocess input
# def preprocess_input(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
#     bag = [1 if word in tokens else 0 for word in words]
#     return np.array([bag])

# # Function to predict class
# def predict_class(text):
#     input_data = preprocess_input(text)
#     prediction = model.predict(input_data)[0]
#     confidence_index = np.argmax(prediction)
#     confidence_score = prediction[confidence_index]
    
#     if confidence_score > 0.5:  # Set confidence threshold
#         return classes[confidence_index]
#     else:
#         return "unknown"

# # Function to generate response
# def get_response(tag):
#     for intent in intents["intents"]:
#         if intent["tag"] == tag:
#             return random.choice(intent["responses"])
#     return "I'm sorry, I don't understand that. Can you rephrase?"

# # Streamlit app
# st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

# st.title("Mental Health Chatbot")
# st.markdown(
#     """
# This chatbot is here to assist with mental health-related inquiries. 
# **Note**: This is not a replacement for professional help. For emergencies, contact a professional immediately.
# """
# )

# # Initialize session state for chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Chat input
# with st.container():
#     user_input = st.text_input("You:", "", placeholder="Type your message here...")

#     if st.button("Send"):
#         if user_input.strip() != "":
#             # Process the user input and get a response
#             tag = predict_class(user_input)
#             response = get_response(tag)
            
#             # Add to chat history
#             st.session_state.chat_history.append(("user", user_input))
#             st.session_state.chat_history.append(("bot", response))

# # Display chat history
# for role, message in st.session_state.chat_history:
#     if role == "user":
#         st.markdown(f"**You:** {message}")
#     elif role == "bot":
#         st.markdown(f"**Chatbot:** {message}")


# import streamlit as st
# import numpy as np
# import pickle
# import json
# from keras.models import load_model
# import nltk
# from nltk.stem import WordNetLemmatizer

# # Load the trained model and necessary files
# model = load_model('chatbot_model.keras')
# with open('model_classes.pkl', 'rb') as f:
#     classes = pickle.load(f)
# with open('model_words.pkl', 'rb') as f:
#     words = pickle.load(f)
# with open('intents.json') as json_file:
#     intents = json.load(json_file)

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Function to preprocess user input
# def preprocess_input(user_input):
#     user_input = nltk.word_tokenize(user_input)
#     user_input = [lemmatizer.lemmatize(word.lower()) for word in user_input]
#     return user_input

# # Function to predict the class of the input
# def predict_class(user_input):
#     input_data = [0] * len(words)
#     for word in user_input:
#         if word in words:
#             input_data[words.index(word)] = 1
#     input_data = np.array(input_data).reshape(1, -1)
#     prediction = model.predict(input_data)
#     return classes[np.argmax(prediction)]

# # Function to get a response based on the predicted class
# def get_response(predicted_class):
#     for intent in intents['intents']:
#         if intent['tag'] == predicted_class:
#             return np.random.choice(intent['responses'])

# # Streamlit app layout
# st.title("Chatbot Application")
# st.write("Welcome to the Chatbot! Type your message below:")

# # User input
# user_input = st.text_input("You: ")

# if st.button("Send"):
#     if user_input:
#         processed_input = preprocess_input(user_input)
#         predicted_class = predict_class(processed_input)
#         response = get_response(predicted_class)
#         st.write(f"Chatbot: {response}")
#     else:
#         st.write("Please enter a message.")



import streamlit as st
import numpy as np
import json
import pickle
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Load the model and data
model = load_model('chatbot_model.keras')
with open('model_classes.pkl', 'rb') as f:
    classes = pickle.load(f)
with open('model_words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('intents.json') as json_file:
    intents = json.load(json_file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(user_input):
    user_input = nltk.word_tokenize(user_input)
    user_input = [lemmatizer.lemmatize(word.lower()) for word in user_input]
    return user_input

# Function to predict the class of the input
def predict_class(user_input):
    input_data = np.zeros(len(words))
    for word in user_input:
        if word in words:
            input_data[words.index(word)] = 1
    prediction = model.predict(np.array([input_data]))[0]
    return classes[np.argmax(prediction)]

# Function to get a response based on the predicted class
def get_response(predicted_class):
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            return np.random.choice(intent['responses'])

# Streamlit app layout
st.title("Elise: Your Personal Mental Health Chatbot!")
st.subheader("Talk to Elise about you want to talk about, without any fear!")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.write(chat)

# User input
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Preprocess input
        processed_input = preprocess_input(user_input)
        
        # Predict class
        predicted_class = predict_class(processed_input)
        
        # Get response
        response = get_response(predicted_class)
        
        # Update chat history
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Elise: {response}")
        
        # Clear input field
        st.rerun()
