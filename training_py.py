import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data (ensure this path is correct)
with open('D:/Download/Health-Care-Chatbot-main/Health-Care-Chatbot-main/intents.json') as json_file:
    intents = json.load(json_file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and process data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize each pattern
        words.extend(word_list)  # Add to the words list
        documents.append((word_list, intent['tag']))  # Save pattern and associated tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add tag to classes if not already added

# Lemmatize words and filter out ignored letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Unique sorted words
classes = sorted(set(classes))  # Unique sorted tags

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)  # Empty output list for one-hot encoding

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create the bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)  # Empty output row for classification
    output_row[classes.index(document[1])] = 1  # Set the correct tag's position to 1
    
    # Add to the training data
    training.append([bag, output_row])

# Shuffle and convert to NumPy arrays
random.shuffle(training)

# Ensure all training data is consistent in shape
training = np.array(training, dtype=object)

# Check the shape of the training array
print(f"Training array shape: {training.shape}")

train_x = np.array(list(training[:, 0]), dtype=np.float32)  # Features (bag of words)
train_y = np.array(list(training[:, 1]), dtype=np.float32)  # Labels (one-hot encoded)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Input layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(64, activation='relu'))  # Hidden layer
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer (softmax for multi-class classification)

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbotmodel.keras')  # Native Keras format
print('Training Done')
