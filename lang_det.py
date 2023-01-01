import pickle
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

this_directory = Path(__file__).parent

le = LabelEncoder()
labels = ['English', 'Russian', "Uzbek"]

le.fit_transform(labels)

with open(this_directory/'src/vectorizer.pickle' , 'rb') as f:
    cv = pickle.load(f)

with open(this_directory/'src/langdet.pickle' , 'rb') as f:
    model = pickle.load(f)

def predict_lang(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     lang
     return lang[0] # returning the language
