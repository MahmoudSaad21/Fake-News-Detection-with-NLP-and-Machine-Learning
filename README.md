# Fake News Detection with NLP and Machine Learning

## 📖 Overview
This project leverages Natural Language Processing (NLP) and Machine Learning techniques to detect fake news. It performs a comprehensive analysis of textual data, including sentiment analysis, topic modeling, and linguistic feature extraction, to classify news articles as either "Fake News" or "Factual News."  

## 🎯 Features
- **Data Preprocessing**: Cleaning, tokenization, stemming, and lemmatization of text.
- **Exploratory Data Analysis (EDA)**: Visualizations of word frequencies, sentiment distributions, and POS tagging.
- **Named Entity Recognition (NER)**: Identification of key entities in fake vs. factual news.
- **Sentiment Analysis**: Using VADER to classify news articles based on sentiment scores.
- **Topic Modeling**: LDA and LSA models to uncover underlying topics in fake news.
- **Classification Models**: Training models (Logistic Regression, SGD, Random Forest, Gradient Boosting, XGBoost) with hyperparameter tuning using GridSearchCV and RandomizedSearchCV.


## 🛠️ Tools and Libraries
- **NLP**: Spacy, NLTK, VADER Sentiment Analysis, Gensim
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Pandas, NumPy, re, string

## 🗂️ Project Structure
```
.
├── fake_news_data.csv            # Dataset containing news articles and labels
├── main_script.py                # Main script with the full pipeline
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies for the project
```

## 🚀 How to Run the Project
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/MahmoudSaad21/Fake-News-Detection-with-NLP-and-Machine-Learning.git
   cd Fake-News-Detection-with-NLP-and-Machine-Learning
   ```

2. **Install Dependencies**  
   Make sure you have Python 3.8 or higher installed. Then, install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**  
   Place the `fake_news_data.csv` file in the root directory of the project.

4. **Run the Main Script**  
   Execute the pipeline to preprocess data, perform analysis, and train models:  
   ```bash
   python main_script.py
   ```

## 📊 Results
### Model Performance Summary
| Model                 | Train Accuracy | Test Accuracy |
|-----------------------|----------------|---------------|
| **XGBoost**           | 0.898551       | 0.950000      |
| Random Forest         | 0.891304       | 0.900000      |
| Logistic Regression   | 0.920290       | 0.883333      |
| Gradient Boosting     | 0.869565       | 0.883333      |
| SGD Classifier        | 0.905797       | 0.800000      |

### Key Insights
- Gradient Boosting achieved the highest accuracy on the test set.
- Sentiment Analysis revealed a higher prevalence of negative sentiment in fake news.
- Named Entity Recognition highlighted differences in the types of entities emphasized in fake vs. factual news.

## 📈 Visualizations
- **POS Tag Distribution**: Top 10 parts-of-speech for fake vs. factual news.
- **Named Entities**: Visualization of most common named entities in fake and factual news.
- **Unigrams and Bigrams**: Most frequent word patterns in the dataset.
- **Topic Modeling**: Coherence scores for LDA models with varying topics.

## 📝 Future Work
- Extend the dataset for better generalizability.
- Incorporate deep learning models like Transformers for improved accuracy.
- Build a web app for real-time news classification.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request.

--- 

Enjoy detecting fake news! 😊 
