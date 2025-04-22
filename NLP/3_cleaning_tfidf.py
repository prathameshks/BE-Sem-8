# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker #pip install pyspellchecker

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Sample text for demonstration
sample_text = "The quick brownn fox jumpd over the lazyy dogs. They were runing in the field."

print("Original text:")
print(sample_text)
print("-" * 50)

# Step 1: Spelling correction
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Remove punctuation for spell checking
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word and spell.unknown([clean_word]):
            correction = spell.correction(clean_word)
            # Replace the original clean word with the correction
            corrected_word = word.replace(clean_word, correction)
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

corrected_text = correct_spelling(sample_text)
print("After spelling correction:")
print(corrected_text)
print("-" * 50)

# Step 2: Tokenization
tokens = word_tokenize(corrected_text)
print("After tokenization:")
print(tokens)
print("-" * 50)

# Step 3: Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("After stop words removal:")
print(filtered_tokens)
print("-" * 50)

# Step 4: Lemmatization
# Get POS tags for better lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # Default to noun

# Get POS tags
pos_tags = nltk.pos_tag(filtered_tokens)

# Lemmatize with POS tags
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = []

for word, tag in pos_tags:
    wordnet_pos = get_wordnet_pos(tag)
    lemmatized_tokens.append(lemmatizer.lemmatize(word, wordnet_pos))

print("After lemmatization:")
print(lemmatized_tokens)
print("-" * 50)

# Combine all steps into one function
def preprocess_text(text):
    # Step 1: Spelling correction
    corrected = correct_spelling(text)
    
    # Step 2: Tokenization
    tokens = word_tokenize(corrected)
    
    # Step 3: Stop words removal
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    
    # Step 4: Lemmatization with POS tagging
    pos_tags = nltk.pos_tag(filtered)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    
    return lemmatized

# Test the combined function
print("Using the combined preprocessing function:")
result = preprocess_text(sample_text)
print(result)
print("-" * 50)

# Show supported languages for stop words
print("Languages supported for stop words:")
print(stopwords.fileids())
print(f"Total English stop words: {len(stopwords.words('english'))}")