# Resume Screener App

This is a Streamlit-based NLP web application that predicts the top 3 job categories of a resume along with confidence scores. You can upload resumes in PDF, DOCX, or TXT format, and the app will extract the text, clean it, and use a pre-trained Support Vector Machine (SVM) model to classify the resume.



## Installation

### Prerequisites
- Python 3.6 or later
- Streamlit
- Scikit Learn


To run this project locally, you need Python 3.6 or later. Follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/MAhad01/resume_screener.git
    cd resume_screener
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the pre-trained model and vectorizer files:**

    Make sure you have the following files in the root directory of your project:
    - `clf.pkl` (the pre-trained SVM model)
    - `tfidf.pkl` (the TF-IDF vectorizer)
    - `encoder.pkl` (the label encoder)

## Usage

To run the app, simply execute the following command:

```bash
streamlit run app.py
```

This will start a local server, and you can access the app in your web browser at `http://localhost:8501`.

### How to Use the App

1. Upload a resume file in PDF, DOCX, or TXT format.
2. Optionally, view the extracted text by selecting the checkbox.
3. The app will display the top 3 predicted job categories with confidence scores in a bar chart.

## Project Structure

- `app.py`: Main script for the Streamlit app.
- `requirements.txt`: List of required Python packages.
- `clf.pkl`: Pre-trained SVM model.
- `tfidf.pkl`: TF-IDF vectorizer.
- `encoder.pkl`: Label encoder for job categories.

## Model Training

The pre-trained model was trained using a dataset of resumes. The following steps outline the training process:

1. **Data Collection**: Collect resumes labeled with job categories.
2. **Text Cleaning**: Clean the text data using regular expressions and text preprocessing techniques.
3. **Vectorization**: Convert the text data into numerical features using TF-IDF vectorization.
4. **Model Training**: Train the SVM model using the processed data.
5. **Evaluation**: Evaluate the model's performance and save the trained model and vectorizer.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a pull request

