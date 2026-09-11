import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.title("RESUME SCREENING SYSTEM")

def extract_text_pdf(upload_file):
        text=""
        with pdfplumber.open(upload_file) as pdf:
            for pages in pdf.pages:
                page_text=pages.extract_text()
                if page_text:
                   text += page_text + "\n"
        return text
    
def extract_text_image(upload_file):
    image=Image.open(upload_file)
    text=pytesseract.image_to_string(image)
    return text


patterns = [
        re.compile(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience'),
        re.compile(r'experience\s*(?:of|:|-)?\s*(\d+)\+?\s*(?:years?|yrs?)'),
        re.compile(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|as|with)'),
        re.compile(r'total\s*experience\s*[:\-]?\s*(\d+)\+?\s*(?:years?|yrs?)'),
        re.compile(r'(\d+)\s*-\s*\d+\s*(?:years?|yrs?)'), 
        re.compile(r'over\s*(\d+)\s*(?:years?|yrs?)'),
        re.compile(r'(\d+)\+?\s*(?:years?|yrs?)')
    ]
def information(text):
    for p in patterns:
        match=re.search(p,text)
        if match:
            return int(match.group(1))
    return 0     


MONTH_NAMES = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)'
data_pattern= re.compile(rf'(?:{MONTH_NAMES}\s+|\d{{1,2}}/)?(\d{{4}})\s*(?:-|–|to)\s*(?:{MONTH_NAMES}\s+|\d{{1,2}}/)?(present|current|\d{{4}})',re.IGNORECASE)
def extract_experience(text):
    match=re.findall(data_pattern,text.lower())
    total_year=0
    for start,end in match:
            start_year=int(start)
            end=end.strip()
            if 'present' in end or 'current' in end:
               end_year = datetime.now().year
            else:
                try:
                  end_year = int(end)
                except ValueError:
                  continue 
            total_year+=end_year-start_year
    return total_year    



def combine_exp(text):
        exp=information(text)
        if exp==0:
           exp=extract_experience(text)
        return exp    


nltk.download("stopwords")
nltk.download("wordnet")

stop_words=set(stopwords.words("english"))
lemmantize=WordNetLemmatizer()

def text_preprocessing(text):
       if text is None:
           return ""
       text=text.lower()
       text=re.sub(r'[^a-zA-Z\s]',"",text)
       text=re.sub(r'https?://\S+',"",text)
       word=text.split()
       words=[lemmantize.lemmatize(w) for w in word if w not in stop_words]
       return " ".join(words)


jd_option=st.radio("How would you like to provide the jd ?",["paste text","upload pdf or image"])
jd_input=""
if jd_option=="paste text":
    jd_input=st.text_area("Paste job description")
elif jd_option=="upload pdf or image":
    jd_file=st.file_uploader("Upload PDF")
    if jd_file:
        jd_file_extension = jd_file.name.split(".")[-1].lower()
        if jd_file_extension == "pdf":
           jd_input = extract_text_pdf(jd_file)
        elif jd_file_extension in ["jpeg","jpg","png"]:
            jd_input=extract_text_image(jd_file) 
        st.success("Job Description extracted sucessfully")

upload_file=st.file_uploader("PASTE OR UPLOAD YOUR RESUME",type=["pdf","jpg","jpeg","png"],accept_multiple_files=True)
if st.button("Rank Resume"):
    if not jd_input:
        st.warning("Please paste or upload your job Description first")
    elif not upload_file:
        st.warning("Pleasse upload your input resume ")    
    else :
        

        resume_text=[]
        resume_name=[]
        experience=[]

        for files in upload_file:
            file_extension=files.name.split(".")[-1].lower()
            if file_extension == "pdf":
                raw_text=extract_text_pdf(files)
            elif file_extension in ["jpeg","jpg","png"]:
                raw_text=extract_text_image(files)
            else:
                raw_text=" "        
            cleaned_text=text_preprocessing(raw_text)
            cleaned_job_des=text_preprocessing(jd_input)
            resume_text.append(cleaned_text)
            resume_name.append(files.name)

            exp=combine_exp(raw_text)
            experience.append(exp)

    tf=TfidfVectorizer()
    corpus=[cleaned_job_des] + resume_text
    result=tf.fit_transform(corpus)

    jd_vector=result[0].reshape(1,-1)
    clean_vector=result[1:]

    similarity=cosine_similarity(jd_vector,clean_vector).flatten()

    res=[]
    for name,score,exp  in zip(resume_name,similarity,experience):
        res.append({
            "Resume":name,
            "Match":score*100,
            "experience":exp })

    res = sorted(res , key=lambda x : x["Match"],reverse=True )
    if res:
        st.subheader("Ranking Summary")
        summary_table=[{
            "Resume": r["Resume"],
            "Match" : r["Match"],
            "Experience (yrs)" : r["experience"]
            }
            for r in res ]
    st.dataframe(summary_table)

    
    name=[r["Resume"] for r in res]
    score=[r["Match"] for r in res]
    fig,axs=plt.subplots()
    axs.barh(name,score)
    st.pyplot(fig)


else:
    st.warning("please provide jd and upload input resume")   

   