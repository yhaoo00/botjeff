#refer https://github.com/jerrytigerxu/Simple-Python-Chatbot/blob/master/chatgui.py
#https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#create data structure to hold context
context = {}

def tokenizer(sentence):
	#tokenize the patterns
	sentence_words = nltk.word_tokenize(sentence)
	#stem and lower each words
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

	return sentence_words

def bag_of_words(sentence, words, show_details=False):
	#tokenize the patterns
	sentence_words = tokenizer(sentence)
	#bag of words
	bag = [0 for _ in range(len(words))]

	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i] = 1
				#show details if set show details to true
				if show_details:
					print("bags: %s" % w)
	
	return(np.array(bag))

def prediction(sentence, model, show_details=False):
	pre = bag_of_words(sentence, words, show_details=True)
	#get probabilities from the model
	res = model.predict(np.array([pre]))[0]
	#set the error threshold as 25%
	SET_ERROR_THRESHOLD = 0.25
	#if probability < error threshold, filter out the predictions
	results = [[i, r] for i, r in enumerate(res) if r > SET_ERROR_THRESHOLD]

	#sort the probability
	results.sort(key=lambda x:x[1], reverse=True)
	return_list = []

	#get highest probability and the intents
	for r in results:
		return_list.append({"intents": classes[r[0]], "probability": str(r[1])})

	#show details if set show details to true
	if show_details:print("Prediction: ", return_list)

	#return tuple of intents and probability
	return return_list

def getResponse(ints, intents_json, show_details=False):
	#set random ID to get context
	ID = '123'
	#get tags from intents
	tag = ints[0]['intents']
	#get intents
	list_of_intents = intents_json['intents']

	#loop if there are matches
	for i in list_of_intents:
		#find matching tag
		if(i['tag'] == tag):
			#set context for this intent if necessary
			if 'context_set' in i:
				#show details if true
				if show_details: print('context: ', i['context_set'])
				context[ID] = i['context_set']

			#check if this intent is contextual and applies to this user's conversation
			if not 'context filter' in i or \
				(ID in context and 'context_filter' in i and i['context_filter'] == context[ID]):
				#show details if true
				if show_details: print('tag: ', i['tag'])

				#give random responses
				result = random.choice(i['responses'])

	#return the message
	return result

def bot_response(msg):
	#get probability
	ints = prediction(msg, model, show_details=True)
	#get response
	response = getResponse(ints, intents, show_details=True)

	#return the message
	return response

#Creating GUI with tkinter
import tkinter
from tkinter import *

#function send onclick()
def send():
	msg = box.get("1.0", 'end-1c').strip()
	box.delete("0.0", END)

	if msg != '':
		log.config(state=NORMAL)
		log.insert(END, "You: " + msg + '\n\n')
		log.config(foreground="#442265", font=("Verdana", 8))

		res = bot_response(msg)
		log.insert(END, "Jeff: " + res + '\n\n')

		log.config(state=DISABLED)
		log.yview(END)

#function send bind <Return>
def sendReturn(event):
	msg = box.get("1.0", 'end-1c').strip()
	box.delete("0.0", END)

	if msg != '':
		log.config(state=NORMAL)
		log.insert(END, "You: " + msg + '\n\n')
		log.config(foreground="#442265", font=("Verdana", 8))

		res = bot_response(msg)
		log.insert(END, "Jeff: " + res + '\n\n')

		log.config(state=DISABLED)
		log.yview(END)

root = Tk()
root.title("Bot Jeff")
root.geometry("500x600")
root.resizable(width=FALSE, height=FALSE)
root.configure(background='lightblue')
root.iconbitmap('jeff.ico')

log = Text(root, bd=0, bg="White", height="11", width="50", font="Arial", padx=10, pady=10)

log.config(state=NORMAL)
log.insert(tkinter.INSERT, "Jeff: Welcome to Matheven Bike Rental Store. I'm BOT Jeff!\n\n")
log.config(foreground="#442265", font=("Verdana", 8))
log.config(state=DISABLED)

scrollbar = Scrollbar(root, command=log.yview, cursor="heart")
log['yscrollcommand'] = scrollbar.set

SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="10", height="2", bd=0, bg="#3F8CFF", activebackground="#1E519C", fg='#ffffff', command= send, activeforeground="#AECFFF")

box = Text(root, bd=0, bg="white", width="29", height="2", font=("Arial", 10))
root.bind('<Return>', sendReturn)

scrollbar.place(x=476,y=6, height=531)
log.place(x=6,y=6, height=531, width=470)
box.place(x=6,y=546, height=45, width=365)
SendButton.place(x=378,y=546, height=45)


root.mainloop()