# Simple English-Hindi Translator

A basic Python script that translates English text to Hindi using a predefined dictionary.
The application has a simple web interface built with Streamlit.

## Features

*   Translates English words/sentences to Hindi.
*   Uses NLTK for word tokenization and lemmatization.
*   Allows adding new words to the dictionary via the UI.
*   Can save the updated dictionary to a `english_hindi_dict.json` file.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (This installs Streamlit and NLTK. The app will try to download NLTK's 'punkt' and 'wordnet' data if missing.)

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open the translator in your web browser.

## Files

*   `app.py`: The Streamlit web application.
*   `translator_backend.py`: The core translation logic.
*   `requirements.txt`: List of Python packages needed.
*   `english_hindi_dict.json` (optional): External dictionary file loaded/saved by the app.

## Note

This is a rule-based translator and its accuracy depends heavily on the provided dictionary. It does not handle complex grammar.