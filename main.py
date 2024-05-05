import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
from textstat import flesch_kincaid_grade
from nltk.corpus import cmudict

# load data set
df = pd.read_excel("Input.xlsx")
print(df)
url = df["URL"]


# scrap data
def scrape_data(url, first_class, head_class, second_class=None):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # empty list for data
        scraped_data = []

        heading = soup.find_all('h1', class_=head_class)
        scraped_data.extend([item.text.strip() for item in heading])

        # Check if the first class exists and has data
        first_data = soup.find_all('div', class_=first_class)
        if first_data:
            # Extract data from the first class and add it to the list
            scraped_data.extend([item.text.strip() for item in first_data])
        elif second_class:  # Check if second_class is provided
            # Extract data from the second class and add it to the list
            second_data = soup.find_all('div', class_=second_class)
            scraped_data.extend([item.text.strip() for item in second_data])
        else:
            print("No data found in specified classes.")

        return scraped_data

    except requests.RequestException as e:
        # Handle request exceptions
        print("Failed to retrieve data from the URL:", e)
        return None
    except Exception as e:
        # Handle other exceptions
        print("An error occurred:", e)
        return None


# define classes

c1 = "td-post-content tagdiv-type"
head_class = ["entry-title", "entry-title"]
c2 = "tdb-block-inner td-fix-index"

# apply the function

text = url.apply(scrape_data, args=(c1, head_class, c2))
print(text)

# convert into dataframe

data = pd.Series(text)


# remove list from value
def remove_list(value):
    if isinstance(value, list):
        return ', '.join(value)
    else:
        return value


# apply function

series_without_lists = data.apply(remove_list)
print(series_without_lists)
data = pd.DataFrame(series_without_lists)

# convert into lower
data["URL"] = data["URL"].str.lower()
data["original_data"] = data["URL"]

# remove a "\n" value with empty string

data['URL'] = data["URL"].str.replace('\n', '')


# Function to remove punctuation
def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    else:
        return text


# apply function

data["URL"] = data["URL"].apply(remove_punctuation)

# Remove Stopword


stop_words = []
stop_words_dir = "StopWords"  # Directory containing stop word text files

encodings = ['utf-8', 'latin-1', 'utf-16']

# Iterate over files and encodings
for filename in os.listdir(stop_words_dir):
    for encoding in encodings:
        try:
            with open(os.path.join(stop_words_dir, filename), 'r', encoding=encoding) as file:
                stop_words.extend(file.read().splitlines())
            break
        except UnicodeDecodeError:
            continue

def filter_stop_words(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    else:
        return text

data['URL'] = data['URL'].apply(filter_stop_words)

# Display the updated DataFrame
print(data)


# positive and negative

positive_words = []
negative_words = []

folder_path = "MasterDictionary"  # Folder containing positive and negative word files

encodings = ['utf-8', 'latin-1', 'utf-16']
for filename in os.listdir(folder_path):
    if filename.endswith("positive-words.txt"):
        for encoding in encodings:
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding=encoding) as file:
                    positive_words.extend(file.read().splitlines())
                # Break the loop if reading succeeds without errors
                break
            except UnicodeDecodeError:
                # Continue to the next encoding if decoding fails
                continue
    elif filename.endswith("negative-words.txt"):
        for encoding in encodings:
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding=encoding) as file:
                    negative_words.extend(file.read().splitlines())
                # Break the loop if reading succeeds without errors
                break
            except UnicodeDecodeError:
                # Continue to the next encoding if decoding fails
                continue



positive_words_filtered = [word for word in positive_words if word.lower() not in stop_words]

negative_words_filtered = [word for word in negative_words if word.lower() not in stop_words]

positive_dict = {word: 1 for word in positive_words_filtered}
negative_dict = {word: -1 for word in negative_words_filtered}

# calculate positive and negative score

def calculate_scores(text):
    if isinstance(text, str):  # Check if text is a string
        # Tokenize the text
        tokens = word_tokenize(text)

        # Calculate positive score
        positive_score = sum(1 for token in tokens if token.lower() in positive_dict)

        # Calculate negative score
        negative_score = sum(1 for token in tokens if token.lower() in negative_dict)

        return positive_score, negative_score
    else:
        return 0, 0  # Return 0 for both scores if text is not a string

# Step 2: Apply the function to the 'URL' column and create new columns for positive score and negative score
data[['Positive Score', 'Negative Score']] = data['URL'].apply(lambda x: pd.Series(calculate_scores(x)))

# Display the updated DataFrame
print("_____________ positive and negative ___________________ ")
print(data[['Positive Score', 'Negative Score']])
print("________________________________________________________")


# Polarity score

data["Polarity Score"] = (data["Positive Score"] - data["Negative Score"])/ (data["Positive Score"] + data["Negative Score"] + 0.000001)

print("_____________ Polarity Score ___________________ ")
print(data["Polarity Score"])
print("________________________________________________ ")

# word count

def count_words(text):
    # Check if text is None
    if text is None:
        return 0
    # Split the text into words
    words = text.split()
    # Return the count of words
    return len(words)

# apply function

data['word_count'] = data['URL'].apply(count_words)

# Subjectivity Score

data['Subjectivity Score'] = (data["Positive Score"] + data["Negative Score"]) / (data["word_count"] + 0.000001)

print("_____________________________Subjectivity Score___________________________________")
print(data["Subjectivity Score"])
print("__________________________________________________________________________________")


# count_sentences

def count_sentences(text):
    if text is None:
        return 0
    else:
        # Tokenize the text into sentences and count the number of sentences
        sentences = nltk.sent_tokenize(text)
        total_sentences = len(sentences)
        return total_sentences

data['Total Sentences'] = data['original_data'].apply(count_sentences)

#Average Sentence Length

data["Average Sentence Length"] = data["word_count"] / data["Total Sentences"]


print("____________________Average Sentence Length__________________")
print(data['Average Sentence Length'])


# complex_word

def calculate_grade_level(text):
    if text is None:
        return 0  # Return a default value if text is None
    else:
        # Calculate the Flesch-Kincaid Grade Level
        grade_level = flesch_kincaid_grade(text)
        return grade_level


data['complex_word'] = data['original_data'].apply(calculate_grade_level)

print(data['complex_word'])


# Fog Index

data["Fog Index"] = 0.4*(data["Average Sentence Length"] + data["complex_word"])

print("____________________fog index__________________")

print(data["Fog Index"])


#Average Number of Words Per Sentence

data["Average Number of Words Per Sentence"] = (data["word_count"].sum() / data["Total Sentences"].sum())

print("____________________Average Number of Words Per Sentence___________________")
print(data["Average Number of Words Per Sentence"])


# Complex Word Count

prondict = cmudict.dict()

# Define a function to count complex words
def count_complex_words(text):
    if text is None:
        return 0

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    tagged_words = nltk.pos_tag(words)

    complex_word_count = sum(1 for word, pos in tagged_words if len(prondict.get(word.lower(), [])) > 2)

    return complex_word_count

data['Complex Word Count'] = data['original_data'].apply(count_complex_words)

print("__________________________Complex Word Count___________________")
print(data['Complex Word Count'])


# Syllable Count Per Word


def count_syllables(word):
    if word is None:
        return 0  # Return 0 if the word is None

    # List of common suffixes to exclude
    exceptions = ["es", "ed"]

    # Remove exceptions from the end of the word
    for suffix in exceptions:
        if word.endswith(suffix):
            word = word[:len(word) - len(suffix)]

    # Count vowels (assuming English language)
    vowels = "aeiou"
    count = 0
    prev_char_was_vowel = False
    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    # Handle special case where word ends with 'e'
    if word.endswith('e') and count > 1:
        count -= 1

    return count

# Apply count_syllables function to each word in the text_column
data['syllable_counts'] = data['URL'].apply(lambda x: [count_syllables(word) if word is not None else 0 for word in str(x).split()])


print(data['syllable_counts'])

#Personal Pronouns

def count_personal_pronouns(text):
    if text is None:
        return 0  # Return 0 if the text is None

    # Define the regex pattern to match personal pronouns
    pattern = r'\b(?:I|we|my|ours|us)\b'

    # Find all matches in the text
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    # Filter out instances where "US" is a match
    matches = [match for match in matches if match.lower() != 'US']

    # Count the occurrences of personal pronouns
    count = len(matches)

    return count

data['personal_pronoun_count'] = data['original_data'].apply(count_personal_pronouns)

print(data['personal_pronoun_count'])

#  Average Word Length

def average_word_length(text):
    if text is None:
        return 0

    # Tokenize the text into words
    words = text.split()

    # Calculate the total number of characters in all words
    total_chars = sum(len(word) for word in words)

    # Calculate the total number of words
    total_words = len(words)

    # Calculate the average word length
    if total_words > 0:
        average_length = total_chars / total_words
    else:
        average_length = 0

    return average_length


data['average_word_length'] = data['original_data'].apply(average_word_length)

print(data['average_word_length'])


data.to_csv('output.csv')