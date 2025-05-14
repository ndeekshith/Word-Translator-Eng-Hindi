import streamlit as st
from translator_backend import EnglishHindiTranslator, download_nltk_data_once, lemmatizer

# --- Page Configuration ---
st.set_page_config(
    page_title="English-Hindi Translator",
    page_icon="🇮🇳",
    layout="wide"
)

def init_nltk_and_lemmatizer():
    print("Attempting NLTK data download (if needed)...")
    download_success = download_nltk_data_once() 
    
    if not download_success:
        # This warning will appear on the Streamlit page
        st.warning(
            "NLTK data download might have had issues. Please check the console output "
            "where you ran `streamlit run`. Translation quality may be affected. "
            "You might need to install 'punkt' and 'wordnet' manually if errors persist: "
            "In Python, run `import nltk; nltk.download('punkt'); nltk.download('wordnet')`."
        )
    else:
        print("NLTK setup appears complete from Streamlit's perspective.")
    
    return download_success

NLTK_READY = init_nltk_and_lemmatizer()



@st.cache_resource
def get_translator():
    print("Initializing translator instance...")
    if not NLTK_READY:
        # This check is a bit redundant if the app stops below, but good for robustness
        st.error("Cannot initialize translator: NLTK resources are not ready. Check console.")
        return None # Or raise an exception
    
    translator_instance = EnglishHindiTranslator() # From translator_backend
    print(f"Translator initialized. Base dict size: {translator_instance.initial_dict_size}, Current: {len(translator_instance.eng_to_hindi_dict)}")
    return translator_instance

# Get the translator instance
if NLTK_READY:
    translator = get_translator()
    if translator is None:
        st.error("Failed to initialize the translator. The app cannot continue.")
        st.stop() # Stop the app if translator couldn't be created
else:
    # This message is more prominent if NLTK_READY is false from the start
    st.error(
        "Core NLTK resources ('punkt', 'wordnet') could not be set up. "
        "The translator cannot function. Please check the console output for errors "
        "when starting Streamlit. You may need to manually install them."
    )
    st.stop() 



st.title(" English to Hindi Translator 🇮🇳")


if translator:
    st.info(f"{translator.extended_dict_message} Current dictionary size: {len(translator.eng_to_hindi_dict)} words.")
    st.caption("Note: This is a basic rule-based translator. It may not handle complex sentences or idioms well.")
col1, col2 = st.columns(2)

with col1:
    st.subheader("➡️ English Input")
    english_sentence = st.text_area("Enter English text:", 
                                    value=" Python is powerful language.", 
                                    height=68,
                                    key="english_input")

with col2:
    st.subheader("⬅️ Hindi Translation")
    hindi_translation_display = st.empty() # Placeholder for dynamic translation update

if translator and english_sentence:
    hindi_translation = translator.translate(english_sentence)
    hindi_translation_display.markdown(f"<p style='font-size: 1.1em; font-family: \"Noto Sans Devanagari\", sans-serif;'>{hindi_translation}</p>", unsafe_allow_html=True)
elif translator: 
    hindi_translation_display.markdown("<p style='font-size: 1.1em; color: grey; font-family: \"Noto Sans Devanagari\", sans-serif;'>Translation will appear here...</p>", unsafe_allow_html=True)
else:
    hindi_translation_display.markdown("<p style='font-size: 1.1em; color: red;'>Translator not available.</p>", unsafe_allow_html=True)


st.markdown("---") # Separator

# --- Dictionary Management (in an expander) ---
if translator:
    with st.expander("⚙️ Dictionary Management"):
        st.write("Manage the translator's dictionary for the current session or save changes.")

        # Add new word
        st.subheader("➕ Add New Word")
        with st.form("add_word_form", clear_on_submit=True):
            new_eng_word = st.text_input("English Word:")
            new_hin_word = st.text_input("Hindi Translation:")
            submitted_add = st.form_submit_button("Add to Dictionary (Session Only)")

            if submitted_add:
                if new_eng_word and new_hin_word:
                    translator.add_word(new_eng_word, new_hin_word)
                    st.success(f"Added '{new_eng_word.lower()}': '{new_hin_word}' to the current session's dictionary.")
                    st.caption(f"New dictionary size: {len(translator.eng_to_hindi_dict)} words.")
                    st.warning("This change is for the current session only. Click 'Save Dictionary' to make it persistent.")
                    st.experimental_rerun()
                else:
                    st.error("Please provide both English and Hindi words.")
        
        # Save dictionary
        st.subheader("💾 Save Dictionary")
        if st.button("Save Current Dictionary to 'english_hindi_dict.json'"):
            success, message = translator.create_extended_dictionary() # Method from backend
            if success:
                st.success(message)
                st.balloons()
            else:
                st.error(message)
            # Rerun to update the dictionary status message at the top
            st.experimental_rerun()


with st.expander("💡 How to use a larger dictionary?"):
    st.markdown("""
    To use a much larger dictionary for potentially better translations:
    1.  **Find a dictionary:**
        *   [Indic Dict - English-Hindi](https://github.com/indic-dict/stardict-english-hindi) (might need conversion from StarDict format to JSON)
    2.  **Prepare the JSON file:**
        *   Ensure it's a flat JSON object where keys are English words (lowercase recommended) and values are their Hindi translations.
        *   Example: `{"apple": "सेब", "banana": "केला", ...}`
    3.  **Save it:** Name the file `english_hindi_dict.json` and place it in the **same directory** as this Streamlit app (`app.py` or `main2.py`) and `translator_backend.py`.
    4.  **Relaunch the app:** The app will automatically try to load it on startup. The message at the top of the page will indicate if it was loaded.

    """)


st.sidebar.header("App Information")
if NLTK_READY:
    st.sidebar.success("NLTK Resources: Ready ✅")
else:
    st.sidebar.error("NLTK Resources: Setup Issue ❌")

if translator:
    st.sidebar.info(f"Current Dictionary Size: {len(translator.eng_to_hindi_dict)}")
else:
    st.sidebar.warning("Dictionary: Not loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with [Streamlit](https://streamlit.io) & NLTK.")
st.sidebar.markdown("This is a basic rule-based translator.")