
# **Data Extraction and Text Analysis**

## **Overview**
This project involves extracting text articles from URLs provided in an input Excel file and performing text analysis to compute specific variables. The results are saved in a structured Excel file matching the given output format.

---

## **Approach**
The solution is divided into two primary tasks:
1. **Data Extraction**:  
   - Articles were scraped from the provided URLs using Python libraries like `requests` and `BeautifulSoup`.
   - Only the article title and main content were extracted, excluding website headers, footers, and other irrelevant sections.
   - Each extracted article was saved in a `.txt` file named after the URL ID from the input file.

2. **Text Analysis**:  
   - Each article was analyzed to compute variables such as:
     - Positive Score
     - Negative Score
     - Polarity Score
     - Subjectivity Score
     - Average Sentence Length
     - Percentage of Complex Words
     - Fog Index
     - Word Count
     - Complex Word Count
     - Syllables per Word
     - Personal Pronouns
     - Average Word Length
   - Libraries like `nltk`, `textblob`, and `syllapy` were used for text processing and analysis.

---

## **Dependencies**
The following Python libraries are required to run the script:
- `pandas`: For reading and writing Excel files.
- `openpyxl`: For handling Excel file formats.
- `nltk`: For natural language processing tasks.
- `textblob`: For polarity and subjectivity computation.
- `syllapy`: For syllable count calculations.
- `requests`: For HTTP requests to fetch web content.
- `beautifulsoup4`: For parsing and extracting HTML content.

Install dependencies using the following command:
```bash
pip install pandas openpyxl nltk textblob syllapy requests beautifulsoup4
```

---

## **Files in the Project**
1. **`solutions.py`**: The main Python script for data extraction and analysis.
2. **`Input.xlsx`**: The input file containing URLs and their corresponding IDs.
3. **`Output Data Structure.xlsx`**: The template for the output format.
4. **`extracted_articles/`**: Directory where extracted article text files are saved.
5. **Output File**: Generated Excel file containing the analysis results.

---

## **How to Run**
1. **Setup**:
   - Ensure all dependencies are installed (see the "Dependencies" section).
   - Place the `Input.xlsx` file in the same directory as the script.

2. **Run the Script**:
   Execute the script using the following command:
   ```bash
   python solutions.py
   ```

3. **Output**:
   - Extracted articles are saved as `.txt` files in the `extracted_articles` directory.
   - The analysis results are saved as an Excel file in the same directory as the script.

---

## **Key Notes**
- **Timeout Handling**: The script includes error handling for URL timeouts. If a URL fails to load, it logs the error and skips to the next URL.
- **NLTK Resource Setup**:
  - Ensure the necessary NLTK resources are downloaded:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

---

