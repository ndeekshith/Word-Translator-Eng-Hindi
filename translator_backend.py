import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Global variable to track NLTK download status
NLTK_DATA_DOWNLOADED = False

def download_nltk_data_once():
    """Download required NLTK resources if not already downloaded in this session."""
    global NLTK_DATA_DOWNLOADED
    if NLTK_DATA_DOWNLOADED:
        # print("NLTK data already checked/downloaded in this session.") # Optional: less console noise
        return True

    resources = ['punkt', 'wordnet']
    all_successful = True
    
    print("Checking/Downloading NLTK resources...")
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}.zip')
            elif resource == 'wordnet':
                nltk.data.find(f'corpora/{resource}.zip') # or just 'corpora/{resource}'
            print(f"NLTK resource '{resource}' already available.")
        except nltk.downloader.DownloadError: # This is the specific exception for missing data
            print(f"NLTK resource '{resource}' not found. Attempting to download...")
            try:
                nltk.download(resource, quiet=True) # quiet=True for less verbose output in Streamlit
                print(f"Successfully downloaded {resource}")
            except Exception as e_dl: # Catch download-specific errors
                print(f"Error downloading NLTK resource '{resource}': {e_dl}")
                all_successful = False
        except Exception as e_find: # Catch other errors during nltk.data.find
             print(f"Error checking for NLTK resource '{resource}': {e_find}. Attempting download...")
             try:
                nltk.download(resource, quiet=True)
                print(f"Successfully downloaded {resource}")
             except Exception as e_dl_fallback:
                print(f"Error downloading NLTK resource '{resource}' after check failed: {e_dl_fallback}")
                all_successful = False

    if all_successful:
        NLTK_DATA_DOWNLOADED = True
        print("NLTK data check/download complete.")
    else:
        print("Some NLTK resources might be missing. Translation quality may be affected.")
    return all_successful


lemmatizer = WordNetLemmatizer()

class EnglishHindiTranslator:
    def __init__(self):
        # Start with a small set of common words
        self.eng_to_hindi_dict = {
            # Numbers
            "zero": "शून्य", "one": "एक", "two": "दो", "three": "तीन", "four": "चार",
            "five": "पांच", "six": "छः", "seven": "सात", "eight": "आठ", "nine": "नौ", "ten": "दस",
            
            # Common verbs
            "is": "है", "am": "हूँ", "are": "हैं", "was": "था", "were": "थे", 
            "go": "जाना", "come": "आना", "eat": "खाना", "drink": "पीना", "sleep": "सोना",
            "walk": "चलना", "run": "दौड़ना", "talk": "बात करना", "read": "पढ़ना", "write": "लिखना",
            "see": "देखना", "hear": "सुनना", "know": "जानना", "think": "सोचना", "feel": "महसूस करना",
            "love": "प्यार करना", "like": "पसंद करना", "want": "चाहना", "need": "आवश्यकता होना",
            "have": "रखना", "give": "देना", "take": "लेना", "make": "बनाना", "do": "करना",
            
            # Pronouns
            "i": "मैं", "me": "मुझे", "my": "मेरा", "mine": "मेरा",
            "you": "तुम", "your": "तुम्हारा", "yours": "तुम्हारा",
            "he": "वह", "him": "उसे", "his": "उसका",
            "she": "वह", "her": "उसकी", "hers": "उसकी",
            "it": "यह", "its": "इसका",
            "we": "हम", "us": "हमें", "our": "हमारा", "ours": "हमारा",
            "they": "वे", "them": "उन्हें", "their": "उनका", "theirs": "उनका",
            
            # Common nouns
            "man": "आदमी", "woman": "औरत", "child": "बच्चा", "boy": "लड़का", "girl": "लड़की",
            "house": "घर", "car": "कार", "book": "किताब", "water": "पानी", "food": "खाना",
            "friend": "दोस्त", "family": "परिवार", "mother": "माँ", "father": "पिता",
            "time": "समय", "day": "दिन", "night": "रात", "morning": "सुबह", "evening": "शाम",
            "school": "स्कूल", "work": "काम", "office": "कार्यालय", "home": "घर",
            
            # Conjunctions and prepositions
            "and": "और", "but": "लेकिन", "or": "या", "if": "अगर", "because": "क्योंकि",
            "with": "के साथ", "without": "के बिना", "in": "में", "on": "पर", "at": "पर",
            "for": "के लिए", "from": "से", "to": "को", "until": "तक", "after": "के बाद",
            "before": "के पहले", "between": "के बीच", "under": "के नीचे", "over": "के ऊपर",
            
            # Adjectives
            "good": "अच्छा", "bad": "बुरा", "big": "बड़ा", "small": "छोटा", "high": "ऊंचा",
            "low": "नीचा", "long": "लंबा", "short": "छोटा", "hot": "गरम", "cold": "ठंडा",
            "new": "नया", "old": "पुराना", "young": "जवान", "happy": "खुश", "sad": "दुखी",
            "beautiful": "सुंदर", "ugly": "बदसूरत", "rich": "अमीर", "poor": "गरीब",
            
            # Questions
            "what": "क्या", "who": "कौन", "where": "कहाँ", "when": "कब", "why": "क्यों", "how": "कैसे",
            
            # Tech words
            "computer": "कंप्यूटर", "internet": "इंटरनेट", "program": "प्रोग्राम",
            "software": "सॉफ्टवेयर", "language": "भाषा", "code": "कोड", "python": "पायथन",
            
            # Misc common words
            "hello": "नमस्ते", "world": "दुनिया", "yes": "हाँ", "no": "नहीं", "please": "कृपया",
            "thank": "धन्यवाद", "sorry": "माफ़ कीजिए", "welcome": "स्वागत है",
            "this": "यह", "that": "वह", "these": "ये", "those": "वे", "here": "यहाँ", "there": "वहाँ",
            "today": "आज", "tomorrow": "कल", "yesterday": "कल", "now": "अब", "later": "बाद में",
            "example": "उदाहरण", "simple": "सरल", "powerful": "शक्तिशाली", "every": "हर", "all": "सब",
            "some": "कुछ", "many": "बहुत"
        }
        self.initial_dict_size = len(self.eng_to_hindi_dict)
        self.extended_dict_message = "" # To store messages for Streamlit UI
        self.load_extended_dictionary()
        
    def load_extended_dictionary(self):
        """Load a larger dictionary from a JSON file if available"""
        dict_file = "english_hindi_dict.json"
        
        if os.path.exists(dict_file):
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    extended_dict = json.load(f)
                
                original_size_before_update = len(self.eng_to_hindi_dict)
                self.eng_to_hindi_dict.update(extended_dict)
                words_added_from_file = len(self.eng_to_hindi_dict) - original_size_before_update
                
                self.extended_dict_message = (f"Loaded {words_added_from_file} new/updated words from '{dict_file}'. "
                                              f"Total words from file: {len(extended_dict)}.")
                print(self.extended_dict_message) # For console logging
            except json.JSONDecodeError as e:
                self.extended_dict_message = f"Error decoding JSON from '{dict_file}': {e}. File might be corrupted."
                print(self.extended_dict_message)
            except Exception as e:
                self.extended_dict_message = f"Error loading extended dictionary '{dict_file}': {e}"
                print(self.extended_dict_message)
        else:
            self.extended_dict_message = f"No extended dictionary ('{dict_file}') found. Using basic dictionary of {self.initial_dict_size} words."
            print(self.extended_dict_message)
        # print(f"Current dictionary has {len(self.eng_to_hindi_dict)} words") # For console logging
    
    def create_extended_dictionary(self, file_path="english_hindi_dict.json"):
        """Creates a JSON file with the current dictionary"""
        try:
         
            dict_to_save = self.eng_to_hindi_dict

            with open(file_path, 'w', encoding='utf-8') as f:
                # Sort dictionary for consistent output, if desired
                sorted_dict = dict(sorted(dict_to_save.items()))
                json.dump(sorted_dict, f, ensure_ascii=False, indent=4)
            message = f"Dictionary successfully saved with {len(sorted_dict)} words to {file_path}"
            print(message)
            return True, message
        except Exception as e:
            message = f"Error saving dictionary: {e}"
            print(message)
            return False, message
    
    def add_word(self, english, hindi):
        """Add a new word to the dictionary for the current session"""
        self.eng_to_hindi_dict[english.lower()] = hindi
        
    def translate(self, english_sentence):
        """Translate an English sentence to Hindi"""
        # Ensure NLTK data (punkt for tokenization) is available
        if not NLTK_DATA_DOWNLOADED:
            # This is a fallback, ideally app.py ensures this before calling translate
            print("Warning: NLTK data might not be fully downloaded. Attempting translation...")
            # You could try to download again here, but it's better handled at app startup
        
        try:
            english_words = word_tokenize(english_sentence.lower())
        except LookupError: # Specifically for missing 'punkt'
            print("NLTK 'punkt' tokenizer not found. Please ensure it's downloaded. Falling back to simple split.")
            english_words = english_sentence.lower().split()
        except Exception as e: # Other tokenization errors
            print(f"Word tokenization failed: {e}. Falling back to simple split.")
            english_words = english_sentence.lower().split()
            
        hindi_translation_words = []

        for word in english_words:
            # Remove punctuation if it's attached to the word
            clean_word = word.strip('.,!?;:"\'()[]{}') # Added more punctuation
            punct = ""
            if len(word) > len(clean_word):
                # Heuristic to get trailing punctuation
                # This is simple; a more robust solution might use regex
                for i in range(len(word) -1, -1, -1):
                    if word[i].isalnum():
                        break
                    punct = word[i] + punct
                if not punct and clean_word != word : # handle leading punctuation if any
                     # find first alphanumeric char to split
                    first_alnum_idx = -1
                    for idx_c, char_c in enumerate(word):
                        if char_c.isalnum():
                            first_alnum_idx = idx_c
                            break
                    if first_alnum_idx > 0: # leading punct
                        punct = word[:first_alnum_idx] # this would be leading, not ideal for append
                        # For simplicity, we'll stick to trailing punctuation logic primarily
                        # or just ensure clean_word is really clean.
                        # Resetting punct if it seems to be leading, as we append it.
                        # A better approach: split word into (leading_punct, core, trailing_punct)
                        punct = word[len(clean_word):] # Revert to simpler trailing punct

            
            # Try direct lookup first
            translation = self.eng_to_hindi_dict.get(clean_word)

            if translation:
                hindi_translation_words.append(translation + punct)
            else:
                # If direct lookup fails, try lemmatized form
                try:
                    lemma_v = lemmatizer.lemmatize(clean_word, pos='v')  # Try as verb
                    lemma_n = lemmatizer.lemmatize(clean_word, pos='n')  # Try as noun
                    lemma_adj = lemmatizer.lemmatize(clean_word, pos='a') # Try as adjective
                    lemma_adv = lemmatizer.lemmatize(clean_word, pos='r') # Try as adverb
                    
                    # Check in order of likelihood or preference
                    translation_lemma = self.eng_to_hindi_dict.get(lemma_v)
                    if not translation_lemma:
                        translation_lemma = self.eng_to_hindi_dict.get(lemma_n)
                    if not translation_lemma and lemma_n != clean_word : # if lemmatizing as noun changed it, try that first
                         translation_lemma = self.eng_to_hindi_dict.get(lemmatizer.lemmatize(clean_word)) # default pos=NOUN

                    if not translation_lemma: # Generic lemmatize (often noun)
                        lemma_generic = lemmatizer.lemmatize(clean_word)
                        if lemma_generic != clean_word: # only lookup if lemmatization changed the word
                            translation_lemma = self.eng_to_hindi_dict.get(lemma_generic)

                    # Fallback to checking other POS if still not found
                    if not translation_lemma and lemma_v != clean_word:
                        translation_lemma = self.eng_to_hindi_dict.get(lemma_v)
                    if not translation_lemma and lemma_adj != clean_word:
                        translation_lemma = self.eng_to_hindi_dict.get(lemma_adj)
                    if not translation_lemma and lemma_adv != clean_word:
                        translation_lemma = self.eng_to_hindi_dict.get(lemma_adv)


                    if translation_lemma:
                        hindi_translation_words.append(translation_lemma + punct)
                    else:
                        # If word is not found in the dictionary even after lemmatization
                        hindi_translation_words.append(f"[{clean_word}]{punct}")
                except LookupError: # Specifically for missing 'wordnet'
                    print("NLTK 'wordnet' lemmatizer data not found. Cannot lemmatize. Please ensure it's downloaded.")
                    hindi_translation_words.append(f"[{clean_word}]{punct}") # Fallback for missing WordNet
                except Exception as e:
                    print(f"Error lemmatizing word '{clean_word}': {e}")
                    hindi_translation_words.append(f"[{clean_word}]{punct}")

        return " ".join(hindi_translation_words)