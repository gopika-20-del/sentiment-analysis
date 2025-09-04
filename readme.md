# Sentiment Analysis on IMDB Movie Reviews

##  Project Overview
This project applies **Sentiment Analysis** on the IMDB movie reviews dataset using **Machine Learning**.  
It predicts whether a given review is **Positive** or **Negative**.  

Techniques used:
- Text preprocessing with NLTK (stopwords removal, cleaning)
- TF-IDF Vectorization (with bigrams)
- Support Vector Machine (LinearSVC)
- Visualization with Matplotlib, Seaborn, and WordCloud

## Features
- Achieves ~**89% accuracy** on test data
- Generates **confusion matrix** and performance report
- Creates **word clouds** for positive and negative reviews
- Allows testing with **custom user reviews**

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
