import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import syllapy
import os

# Ensure you download NLTK stopwords if not already done
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Step 1: Read Input File
input_file = "Input.xlsx"
df = pd.read_excel(input_file)

# Create output directories
os.makedirs("ExtractedArticles", exist_ok=True)
os.makedirs("ProcessedOutput", exist_ok=True)

# Step 2: Extract Articles
def extract_article(url):
    """
    Extract the article title and text from the given URL.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title').text.strip() if soup.find('title') else "No Title Found"
        
        # Extract main content
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)
        return title, content
    except Exception as e:
        print(f"Error extracting URL: {url}\nError: {e}")
        return None, None

# Extract articles and save to text files
for _, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    title, content = extract_article(url)
    
    if title and content:
        with open(f"ExtractedArticles/{url_id}.txt", "w", encoding="utf-8") as file:
            file.write(title + "\n" + content)

# Step 3: Perform Text Analysis
def count_syllables(word):
    """
    Count the syllables in a word using the syllapy library.
    """
    return syllapy.count(word)

def count_complex_words(words):
    """
    Count words with more than 2 syllables.
    """
    return sum(1 for word in words if count_syllables(word) > 2)

def analyze_text(text):
    """
    Perform text analysis on the given text and compute required metrics.
    """
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Remove stopwords and punctuation
    words_cleaned = [word for word in words if word.isalpha()]
    
    # Count Metrics
    num_sentences = len(sentences)
    num_words = len(words_cleaned)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    complex_word_count = count_complex_words(words_cleaned)
    percentage_complex_words = (complex_word_count / num_words) * 100 if num_words > 0 else 0
    
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    syllables_per_word = sum(count_syllables(word) for word in words_cleaned) / num_words if num_words > 0 else 0
    avg_word_length = sum(len(word) for word in words_cleaned) / num_words if num_words > 0 else 0
    
    # Sentiment Analysis
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    
    # Personal Pronouns
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us|me|mine)\b', text, flags=re.I))
    
    return {
        "Positive Score": max(0, polarity_score),
        "Negative Score": min(0, polarity_score),
        "Polarity Score": polarity_score,
        "Subjectivity Score": subjectivity_score,
        "Avg Sentence Length": avg_sentence_length,
        "Percentage of Complex Words": percentage_complex_words,
        "Fog Index": fog_index,
        "Complex Word Count": complex_word_count,
        "Word Count": num_words,
        "Syllables Per Word": syllables_per_word,
        "Personal Pronouns": personal_pronouns,
        "Avg Word Length": avg_word_length
    }

# Prepare output data
output_data = []

for _, row in df.iterrows():
    url_id = row['URL_ID']
    file_path = f"ExtractedArticles/{url_id}.txt"
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            analysis_results = analyze_text(text)
            output_data.append({**row.to_dict(), **analysis_results})

# Step 4: Save Output to Excel
output_df = pd.DataFrame(output_data)
output_df.to_excel("ProcessedOutput/Output.xlsx", index=False)

print("Processing completed. Output saved in 'ProcessedOutput/Output.xlsx'.")
