import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("extracted_archive/dataset.csv")  # Update path if needed

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean each text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Preprocess 'generated' column
df['clean_text'] = df['generated'].astype(str).apply(preprocess)

# Generate word clouds by category
categories = df['category'].unique()

for cat in categories:
    cat_text = ' '.join(df[df['category'] == cat]['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(cat_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {cat}")
    plt.show()
