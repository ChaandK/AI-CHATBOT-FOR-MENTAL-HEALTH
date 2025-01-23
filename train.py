###ENVIRONMENT SETUP
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
print(sys.executable)

#IMPORTING LIBRARIES
import pickle    #for serializing python objects (Eg- Saving models)
import random    #used to get random numbers
import json      #for working with JSON files
import re
import numpy as np    #for numerical operations
import nltk           #for tokenizing and lemmatizing text
from nltk.stem import WordNetLemmatizer  
import matplotlib.pyplot as plt
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')'

#tensorflow.keras is used here to build DL models, and specifically using Keras, a high level API for  building neural networks within TF
from tensorflow.keras.models import Sequential          #Sequential is a special case of model where model is purely a stck of single-input, single-output layers
from tensorflow.keras.layers import Dense, Activation, Dropout      #Dense (Basic building block where every input connects to every output)
#Activation func. (Decides if a neuron should be activated or not, making the network smart)
#Dropout (Randomly ignores some neurons during training to make the network better at generalizing)

from tensorflow.keras.optimizers import Adam   #for Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping,  ReduceLROnPlateau


###TOKENIZATION FUNCTION
def tokenize(token):
    return nltk.word_tokenize(token)
tokenize("why is this not working?")

lemmatizer = WordNetLemmatizer()  #converts into base or root form [Eg- 'Running' to 'Run]

intents = json.load(open('intents.json'))  #loads JSON File

words = []   #to store all words found in the training patterns
classes = []  #to store unique intent tags (categories) [Eg- 'greeting' or 'neutral-response']
documents = []  #a list of tuples where each tuple contains a tokenized sentence (word list) and its associated intent tag
Ignore_Symbols = [',', '.', '?', '!']   #Specifies characters to ignore

for intent in intents['intents']:   #to go through all the intents in the json file
    tag = intent['tag']
    for pattern in intent['patterns']:   #to go through all the patterns in the intents
        
        #Pre-clean the pattern (remove special characters)
        cleaned_pattern = re.sub(r"[^\w\s]", "", pattern)
        
        word_list = nltk.word_tokenize(pattern)            #splits the sentences into individual words
        
        #tokenize and lemmatize each word in the pattern
        words_list = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(pattern) if word not in Ignore_Symbols] #Each word is being lemmatize while ignoring symbols defined in Ignore_Symbols
        
        #Add words and documents
        words.extend(word_list)          #Add the tokenized words to the words list
        documents.append((word_list, tag))      #Appends the tokenized pattern and its tag as a tuple to documents

        #Add the tag to the classes if its not already present
        if tag not in classes:
            classes.append(tag)

#Removes duplicates by converting to a set, then back to a list, to ensure that the entries are unique and unordered
words = sorted(set(words))       #sorted() - converts the sets back to lists and sorts them
classes = sorted(set(classes))

print(f"Total words: {len(words)}")
print(f"Total classes: {len(classes)}")
print(f"Documents: {len(documents)}")

#Saving data with pickle; saves the words and classes lists to files using pickle module
#It enables you to load them later without reprocessing
# pickle.dump(words, open('model_words.pkl', 'wb'))
# pickle.dump(classes, open('model_classes.pkl', 'wb'))

# During training
with open("model_words.pkl", "wb") as f:
    pickle.dump(words, f)

with open("model_classes.pkl", "wb") as f:
    pickle.dump(classes, f)


#Preparing training data; 
training = []    #will store input-output pairs for training the model
output_empty = [0] * len(classes)       #creates a list of zeroes with equal length to the number of unique classes; this serves as a template for the output
##Focuses on converting the textual input and output into numeric format for ML algo
for document in documents:         #Iterates through each document in the documents list (Document is a tuple containing a tokenized sentence and its associated tag)
    bag = []        #Initialized to store an empty list for each bag of words
    word_patterns = document[0]         #Get the tokenized words from the documents
    word_patterns_set = set(word_patterns)     #Converts the list of words into a set for faster lookup

    for word in words:          #For each word in the words list (contains all unique words from training dataset), it checks if that word is present in word_patterns
        bag.append(1) if word in word_patterns else bag.append(0)       #It appends '1' if the word is present and '0' if not; creating binary representation of whether each word exists in the current document
         
    output_row = list(output_empty)         #create an output row initialized to zeroes; which is of same length as the number of unique classes (intents)
    output_row[classes.index(document[1])] = 1      #The index in output_row that corresponds to the current document's intent tag (document[1]) is set to '1'; Indicates the this particular output(class) is active
    training.append([bag, output_row])      #Append the bag and output row pair to the training list; Each entry in 'training' will be a pair of input features(bag of words) and the corresponding output (the intent representation)

def preprocess_input(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    bag = [1 if word in tokens else 0 for word in words]
    print(f"User input tokens: {tokens}")  # Debugging
    print(f"Bag of words vector: {bag}")   # Debugging
    return np.array([bag])

random.shuffle(training)        #training list is shuffled to mix the order of samples; helps in preventing model from learning the sequence of training samples
training = np.array(training, dtype=object)   #Convert the list to numpy array for easier manipulation; use dtype=object for variable length arrays
train_x = np.array(list(training[:, 0]))     #Extracts the input features (bag of words) from training array and converts it to the list
train_y = np.array(list(training[:, 1]))      #Extracts the output labels (one-hot encoded intents) from training array and converts it into list

## ONE-HOT ENCODING - method for converting categorical variables into binary format (0s and 1s, where '1' indicates the presence of that category and '0' indicates its absence)

print(f"Shape of train_x: {train_x.shape}")
print(f"Shape of train_y: {train_y.shape}")

model = Sequential()    #initializes a sequential model, which is a type of neural network where the layers are stacked sequentially
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))    #Dense(128) - adds a fully connected (dense) layer with 128 neurons(units). Each neuron will compute its output based on its inputs through weights and biases.
#input_shape=(len(train_x[0]),) - defines the input shape for the first layer. len(train_x[0]) indicates the number of features in the i/p data, ensuring the first layer knows how many inputs it will receive(the dimension of the feature vector).
#activation='relu' - specifies the activation function for the layer.
#The rectified Linear unit (ReLU) activation function is used because it helps mitigate issues like vanishing gradients and allows the network to learn complex patterns. It outputs the input directly if its positive; otherwise it outputs zero.
model.add(Dropout(0.3)) #This layer randomly sets 30% of the inputs to zero during training.
#Dropout is a regularization technique that helps prevent overfitting by ensuring that the model does not rely too heavily on any particular neuron.
model.add(Dense(64, activation='relu')) #Another dense layer with 64 neurons is added, again using ReLU activation function. This layer will learn complex representations and patterns from the input data.
model.add(Dropout(0.3))     #this adds regularization again to help reduce overfitting by randomly setting 30% of the activations to zero in the layer preceding it.
model.add(Dense(len(train_y[0]), activation='softmax'))     #this is the output layer of the network.
#The number of neurons in this layer equals the number of unique classes(intents), which corresponds to the length of train_y[0](the one-hot encoding o/p)
#activation='softmax' - used in output for multi-class classification problems. It converts the output into a probability distribution; each output value represents the relative likelihood of each class.
optimizer = Adam(learning_rate=0.001)   #Adam stands for Adaptive Moment Estimation. It adjusts the learning rate of parameters individually based on estimates of first and second moments of gradients.
#learning_rate=0.001 - sets the learning rate for the optimizer. It controls how much the model updates its parameters during training.
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]) #compiles the model
#loss="categorical_crossentropy" - specifies the loss function to be used for training. Categorical cross-entropy is suitable for multi class classification problems, where the target outcomes are one-hot encoded.
#optimizer=optimizer - sets the optimizer defined earlier to be used during training.
#metrics=["accuracy"] - performance measure during training and validation. This will track accuracy through out the training process.
early_stopping = EarlyStopping(monitor = 'loss', patience=10, restore_best_weights=True)    #this callback is used to stop training a model early if it detects that the model's performance on the training data has stopped moving.
#monitor = 'loss' - monitors the loss value during training.
#patience=10 - number of epochs with no improvement after which training will be stopped.
#restore_best_weights=True - restores the model weights to the best epoch based on the monitored metric.
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5)       #This callback reduces the learning rate of the optimizer when the monitored metric stops improving, which can help the model converge more effectively.
# #monitor="loss" - monitors the loss value during training.
# #factor=0.5 - factor by which the learning rate will be reduced.
# #patience=5 - number of epochs with no improvement after which learning rate will be reduced.
history = model.fit(
    np.array(train_x), 
    np.array(train_y), 
    epochs=15, 
    batch_size=32, 
    verbose=1,
    callbacks=[early_stopping, reduce_lr]) #fitting the model
#model.fit() - this methods starts the training process
#np.array(train_x), np.array(train_y) - converts the training input features and corresponding labels into a numpy arrays(if not already in that format).
#epochs=50 - the number of times the entire training dataset will pass through the model.
#batch_size=5 - number of samples per gradient update. smaller batch provides regualar updates but can lead to more noisy gradients.
#verbose=1 - this controls the verbosity of the output during training. Setting it to 1 shows progress updates.

train_loss, train_accuracy = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)

print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Training Loss: {train_loss:.4f}")

# test_loss, test_accuracy = model.evaluate(np.array(test_x), np.array(test_y), verbose=0)
# print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
# print(f"Final Test Loss: {test_loss:.4f}")

model.save('chatbot_model.keras') #Saves the trained model to the specifies file path. the model can later be reloaded to make predictions without needing to retain.

with open("model_CB.pkl", "wb") as f:
    pickle.dump(model, f)

plt.figure(figsize=(12, 5))     #creates a new figure(plot canvas) with a size of 12X5 inches
# Plot Loss
plt.subplot(1, 2, 1)  #Sets up the first subplot in a 1-row, 2-column layout i.e, there will be two side-by-side plots, and this is the first one
#plotting the loss accuracy over epochs
plt.plot(history.history['loss'], label='Training Loss', color='blue')      #history['loss'] contains the accuracy values stored during training
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
# Plot Accuracy
plt.subplot(1, 2, 2)        #Sets up the second subplot in a 1-row, 2-column layout
#plotting the training accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()
# Display the plots
plt.tight_layout()
plt.show()