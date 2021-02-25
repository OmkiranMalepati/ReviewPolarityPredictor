import streamlit as st
import nltk
import base64
import sklearn
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english'))
snostem = nltk.stem.SnowballStemmer('english')

model=pkl.load(open("AmznLR.pkl","rb"))
model1=pkl.load(open("BoW.pkl","rb"))


#re html tags and punc funcs
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',sentence)
    return cleantext
#sentence = 'I am <> Om'
#cleanhtml(sentence)

#func for removing punctuations
import re
def cleanpunc(sentence):
    cleaned = re.sub('[?|!|\'|"|#]',' ',sentence)
    cleaned = re.sub('[.|,|(|)|\|/]',' ',cleaned)
    return cleaned


st.set_page_config(page_title="Review Classifier",page_icon="⚕️",layout="centered",initial_sidebar_state="expanded")


def preprocess(review):   
    
    #Total processing
    def text_preprocessing(final): 
        i=0
        str=''
        s=''
        filtered_sentence=[]
        final = cleanhtml(final)
        for w in final.split():
            for cleaned_words in cleanpunc(w).split(): #after cleanpunc again one word may become two or multiple so they'll be split and stored as different words
                if((cleaned_words.isalpha()&(len(cleaned_words)>2))):
                    if(cleaned_words.lower() not in stop):
                        s = (snostem.stem(cleaned_words.lower()).encode('utf8'))
                        filtered_sentence.append(s)
            str = b" ".join(filtered_sentence)
        return str


    # Pre-processing user input   
    rev=model1.transform([text_preprocessing(review)])

    prediction = model.predict(rev)

    return prediction

    

       
    # front end elements of the web page 
html_temp = """ 
    <div id=1 style ="background-color:orange;padding:10px"> 
    <h1 style ="color:blue;text-align:center;">Review Polarity Predictor</h1> 
    </div>
    
    <style>
    body {
    background-image: url("https://wallpapershome.com/images/pages/pic_h/16055.jpg");
    background-size: cover;
    }
    </style>
    """

html1 = """<h2 style ="color:blue;text-align:right;"> - OM AND SIVA</h2> """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.markdown(html1, unsafe_allow_html = True)
st.markdown("""<h2 style ="color:blue;text-align:left;">Enter your review below</h2> """, unsafe_allow_html = True)
      
# following lines create boxes in which user can enter data required to make prediction
review=st.text_area("")



#user_input=preprocess(sex,cp,exang, fbs, slope, thal )
pred=preprocess(review)




if st.button("Predict"):    
  st.success(pred)
    
   



st.sidebar.subheader("About App")

st.sidebar.info("This web app is to predict polarity of an user review")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check the same")
st.sidebar.info("Don't forget to rate this app")



feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Thank you for rating the app!")
  st.info("Caution: This is just a prediction") 
