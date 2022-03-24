##refer https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/chatgui.py
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import random

import matplotlib.pyplot as plt

#open intents.json and load it
with open("intents.json") as file:
	data = json.load(file)

words = []
classes = []
docs = []
ignore_words = ['?', '!', '.']

#loop each sentences in patterns
for intent in data["intents"]:
	for pattern in intent["patterns"]:

		#take every words and tokenize it
		#who are you --> [who], [are], [you]
		w = nltk.word_tokenize(pattern)
		#add it to word list
		words.extend(w)

		#add to docs in corpus
		docs.append((w, intent["tag"]))

	#add to class list
	if intent["tag"] not in classes:
		classes.append(intent["tag"])

#stem and lower the words, then remove duplicates
#Powered --> power
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
#sort the words
words = sorted(list(set(words)))

#remove duplicates
classes = sorted(list(set(classes)))

#create pickle file for words and classes
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#create training data
training = []

#empty array for output
output_empty = [0 for _ in range(len(classes))]

for doc in docs:
	#initialize bag of words
	bag = []

	#lists of tokenized words in patterns
	current_words = doc[0]
	#stem the words
	current_words = [lemmatizer.lemmatize(word.lower()) for word in current_words]

	#create bag of words array
	#one-hot encoding
	for w in words:
		if w in current_words:
			bag.append(1) #set to 1 if exist
		else:
			bag.append(0) #0 if not exist

	#look throught classes[] and set current tag = 1
	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1

	training.append([bag, output_row])

#shuffle the data and turn into numpy array
random.shuffle(training)
training = np.array(training)

#create train and test lists
#X for patterns
#Y for intents
train_x = list(training[:,0])
train_y = list(training[:,1])

#Model with 3 layers
model = Sequential()
#create 128 neurons for first layers
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
#create 64 neurons for second layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#num of intents as number of neurons to predict output intent
#use softmax activation to get probability
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile the model. 
#After doing some research, I decided to use SGD with Nesterov over Adam.
#refer https://ruder.io/optimizing-gradient-descent/
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fit and save the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=8000, batch_size=32, verbose=1)
model.save('chatbot_model.h5', hist)

'''
print(hist.history.keys())

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

#summarize history for loss
plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
'''
