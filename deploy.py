#Import Required Python Libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
from pickle import load
import re
import string
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer,word_tokenize
import pickle
from pickle import load
import streamlit as st
import rake_nltk
from rake_nltk import Rake

# Body of the application
st.header("Hotel Rating Classification")
st.markdown("This application has been trained on machine learning model - **Support Vector Machines**.")
st.markdown("This application can predict if the given **Review** is **Positive, Negative or Neutral**")

#Input your review for prediction
input_review = (st.text_area("Type your review here...", """"""))

#Loading Both SVM and TfidfVectorizer Intelligence for deployment
svm_deploy = load(open("C:/Users/836916.INDIA/Documents/Python/project/model/svm_deploy.pkl", "rb"))
tf_idf_deploy = load(open("C:/Users/836916.INDIA/Documents/Python/project/model/tf_idf_deploy.pkl", "rb"))

#Initializing necessary functions and stop words
lemmatizer = WordNetLemmatizer()
w_tokenizer=WhitespaceTokenizer()
stop_words = pd.read_csv('C:/Users/836916.INDIA/Documents/Python/project/model/stop.txt',header=None,squeeze=True)

#Text cleaning Functionality
def text_clean(text):
  text=text.lower()
  text=re.sub("\d+","",text) #Remove Numbers
  text=re.sub("\[.*?\]","",text) #Remove text between square brackets
  text=re.sub("\S*https?:\S*","",text) #Remove URLs
  text=re.sub("\S*www?.\S*","",text) #Remove URLs
  text=re.sub("[%s]" % re.escape(string.punctuation),"",text) #Remove All Punctuations
  text=re.sub("\n","",text) #Remove newline space
  text=re.sub(' +', " ", text) #Remove Additional space
  text=text.split() #Split the text into list of words, i.e. tokenization
  text=[word for word in text if word not in list(stop_words)] #Remove stop words
  text=' '.join(text) #join list back to string
  return text
cleaned_text=text_clean(input_review)

#Lemmatization Functionality
def lemmatize(txt):
  list_review=[lemmatizer.lemmatize(word=word, pos=tag[0].lower()) if tag[0].lower() in ['a','r','n','v'] else word for word, tag in pos_tag(word_tokenize(txt))]
  return (' '.join([x for x in list_review if x]))

#transform text into numerical
X=tf_idf_deploy.transform([lemmatize(cleaned_text)])

# Making prediction
if st.button("Click to make prediction"):
    prediction = int(svm_deploy.predict(X)[0])
    if prediction == 0:
        st.error("This is a Negative Review!")
    elif prediction == 1:
        st.warning("This is a Neutral Review!")
    else:
        st.success("This is a Positive Review!")

# Getting Keywords using Rake Module
def get_keywords(text):
    r = Rake(stopwords=set(stop_words), punctuations=set(string.punctuation), include_repeated_phrases=False)
    r.extract_keywords_from_text(input_review)
    words = [re.sub("[%s]" % re.escape(string.punctuation), "", x) for x in r.get_ranked_phrases()]
    words = [x.strip() for x in words if x]
    return words

st.subheader("Influencing Attributes for the Review")

radio=st.sidebar.radio("Click below to get top Keywords!",("Top 10","Top 20","All"))

result = get_keywords(input_review)

if radio=="Top 10":
    for word in result[:10]:
        st.markdown(word)
elif radio=="Top 20":
    for word in result[:20]:
        st.markdown(word)
else:
    for word in result:
        st.markdown(word)