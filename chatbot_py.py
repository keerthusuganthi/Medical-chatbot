import nltk
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load necessary files (ensure paths are correct)
chatbot_model = load_model('chatbotmodel.keras')  # Load chatbot model
with open('D:\Download\Health-Care-Chatbot-main\words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('D:\Download\Health-Care-Chatbot-main\classes.pkl', 'rb') as file:
    classes = pickle.load(file)
with open('D:\Download\Health-Care-Chatbot-main\Health-Care-Chatbot-main\intents.json') as file:
    intents = json.load(file)

# Load the heart disease prediction model and scaler
with open('D:\Download\Health-Care-Chatbot-main\heart_disease_model.pkl', 'rb') as file:
    heart_disease_model = pickle.load(file)
with open('D:\Download\Health-Care-Chatbot-main\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to clean up and lemmatize the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert sentence to bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of the sentence
def predict_class(sentence):
    bow_input = bow(sentence, words, show_details=False)
    prediction = chatbot_model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Heart disease prediction function
def heart_disease_prediction(input_data):
    # Assume input_data is a dictionary of features (e.g., {'age': 50, 'chol': 200, ...})
    data_df = pd.DataFrame([input_data])
    scaled_data = scaler.transform(data_df)
    prediction = heart_disease_model.predict(scaled_data)
    return "High risk of heart disease." if prediction[0] == 1 else "Low risk of heart disease."

# Function to parse heart disease input
def parse_heart_disease_input(input_text):
    # Assuming the input text is space-separated, e.g., "50 1 3 130 200 ..."
    input_list = list(map(float, input_text.split()))
    
    # Map the input values to a dictionary
    return {
        'age': input_list[0],
        'sex': input_list[1],
        'cp': input_list[2],
        'trestbps': input_list[3],
        'chol': input_list[4],
        'fbs': input_list[5],
        'restecg': input_list[6],
        'thalach': input_list[7],
        'exang': input_list[8],
        'oldpeak': input_list[9],
        'slope': input_list[10],
        'ca': input_list[11],
        'thal': input_list[12]
    }

# Main function to get the chatbot response
def chatbot_response(text):
    intents_list = predict_class(text)  # Get predicted intents
    if intents_list and intents_list[0]['intent'] == "heart_disease_assessment":
        # Trigger heart disease prediction if intent matches
        print("Heart disease assessment triggered")
        
        # Parse the input for heart disease data
        user_health_data = parse_heart_disease_input(text)  # This function will map the input correctly
        response = heart_disease_prediction(user_health_data)
    else:
        response = get_response(intents_list, intents)  # Get a normal response (e.g., symptoms or greetings)
    return response

# Example usage
if __name__ == "__main__":
    # Test the chatbot with a heart disease assessment request
    user_input = "50 1 3 130 200 1 0 0 150 0 2.3 1 0 2"  # Example input for heart disease prediction
    print(chatbot_response(user_input))  # This should trigger the heart disease prediction
