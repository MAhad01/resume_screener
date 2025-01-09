import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
import pandas as pd
from scipy.special import expit

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\\S+\\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the top 3 categories of a resume with confidence scores
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    raw_scores = svc_model.decision_function(vectorized_text)
    probabilities = expit(raw_scores)  # Apply sigmoid to get probabilities
    top3_indices = np.argsort(probabilities[0])[-3:][::-1]
    top3_categories = le.inverse_transform(top3_indices)
    top3_confidences = probabilities[0][top3_indices]

    return list(zip(top3_categories, top3_confidences))


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the top 3 job categories with confidence scores displayed graphically.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Top 3 Predicted Categories")
            top3_predictions = pred(resume_text)
            top_category, top_confidence = top3_predictions[0]

            st.markdown(f"**Top Category: {top_category}** with confidence: **{top_confidence:.2f}**")

            # Prepare data for bar chart
            categories = [item[0] for item in top3_predictions]
            confidences = [item[1] for item in top3_predictions]

            chart_data = pd.DataFrame({
                'Category': categories,
                'Confidence': confidences
            })

            st.bar_chart(chart_data.set_index('Category'))

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
