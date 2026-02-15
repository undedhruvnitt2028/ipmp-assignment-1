# Twitter/Social Media Sentiment Analysis Dashboard
## Assignment 1: Foundation Module

A comprehensive real-time sentiment analysis system for social media data (Twitter/X) with text preprocessing, exploratory data analysis, and model selection documentation.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a complete sentiment analysis pipeline for social media data, including:
1. **Data Acquisition**: Fetch 500+ records using APIs or web scraping
2. **Preprocessing**: Clean and normalize text using Regex and NLTK
3. **EDA**: Generate word clouds and frequency distributions
4. **Model Selection**: Comprehensive BERT vs VADER comparison

---

## ✨ Features

### Data Acquisition
- ✅ Multiple data sources (Twitter API, snscrape, sample data)
- ✅ Configurable keyword-based filtering
- ✅ Export to CSV and JSON formats
- ✅ Metadata collection (likes, retweets, timestamps)

### Text Preprocessing
- ✅ URL and mention removal
- ✅ Special character cleaning
- ✅ Stop word removal (with social media extensions)
- ✅ Lemmatization and stemming
- ✅ Emoji and hashtag handling
- ✅ Contraction expansion

### Exploratory Data Analysis
- ✅ Word cloud generation
- ✅ Top 20 word frequency distribution
- ✅ Horizontal bar charts
- ✅ Token length distribution
- ✅ Word length analysis
- ✅ Summary statistics

### Model Documentation
- ✅ Detailed BERT vs VADER comparison
- ✅ Performance metrics and benchmarks
- ✅ Cost analysis
- ✅ Implementation recommendations
- ✅ Code examples

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download Project
```bash
# If you have the files, navigate to the project directory
cd sentiment-analysis-dashboard
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt --break-system-packages

# If you encounter issues, install key packages individually:
pip install pandas nltk matplotlib seaborn wordcloud vaderSentiment --break-system-packages
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Optional: Twitter API Setup
If you want to use real Twitter data:

1. **Get Twitter API credentials**: Visit https://developer.twitter.com
2. **Set environment variable**:
   ```bash
   export TWITTER_BEARER_TOKEN='your_bearer_token_here'
   ```

Alternatively, the system will generate sample data automatically if no API is available.

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline
```bash
# Run everything with one command
python run_complete_pipeline.py
```

### Option 2: Step-by-Step Execution
```bash
# Step 1: Acquire data (500+ records)
python data_acquisition.py

# Step 2: Preprocess text
python preprocessing.py

# Step 3: Generate visualizations
python eda_analysis.py
```

### Option 3: Custom Configuration
Edit the scripts to customize:
- `KEYWORD`: Change search term (default: "Python programming")
- `NUM_RECORDS`: Adjust number of records (default: 500)
- Visualization parameters (colors, sizes, etc.)

---

## 📁 Project Structure

```
sentiment-analysis-dashboard/
│
├── data_acquisition.py          # Script 1: Data collection
├── preprocessing.py              # Script 2: Text preprocessing
├── eda_analysis.py               # Script 3: EDA & visualizations
├── run_complete_pipeline.py     # Master script (runs all)
│
├── BERT_vs_VADER_Analysis.md    # Model comparison document
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── raw_data.csv                 # Output: Raw collected data
├── raw_data.json                # Output: Raw data (JSON format)
├── processed_data.csv           # Output: Cleaned data
│
└── visualizations/              # Output: Generated plots
    ├── wordcloud.png
    ├── frequency_distribution.png
    ├── frequency_horizontal.png
    ├── token_length_distribution.png
    └── word_length_distribution.png
```

---

## 📖 Usage Guide

### 1. Data Acquisition

The `data_acquisition.py` script tries multiple methods in order:
1. **snscrape** (no API key needed)
2. **Twitter API v2** (if bearer token available)
3. **Sample data generation** (fallback)

**Configuration**:
```python
# In data_acquisition.py
KEYWORD = "Python programming"  # Your search term
NUM_RECORDS = 500               # Number of records to collect
```

**Output**:
- `raw_data.csv`: Tabular format with all fields
- `raw_data.json`: JSON format for complex processing

**Sample Output Structure**:
```
id, date, text, username, likes, retweets, replies, source
```

---

### 2. Text Preprocessing

The `preprocessing.py` script applies a comprehensive cleaning pipeline:

**Pipeline Steps**:
1. Convert to lowercase
2. Expand contractions (don't → do not)
3. Remove URLs
4. Remove @mentions
5. Handle hashtags
6. Remove emojis
7. Remove special characters
8. Remove numbers
9. Tokenize
10. Remove stop words
11. Lemmatize/stem
12. Filter short tokens

**Configuration**:
```python
preprocessor = TextPreprocessor(
    use_lemmatization=True,    # Use lemmatization (vs stemming)
    remove_stopwords=True      # Remove common words
)
```

**Output**:
- `processed_data.csv`: Original data + cleaned_text + tokens columns

**Example**:
```
Original: "I LOVE Python programming!!! 😍 #coding https://example.com"
Cleaned:  "love python programming coding"
Tokens:   ["love", "python", "programming", "coding"]
```

---

### 3. Exploratory Data Analysis

The `eda_analysis.py` script generates 5 visualizations:

#### 3.1 Word Cloud
- Visualizes most frequent words
- Size indicates frequency
- Customizable colors and themes

#### 3.2 Frequency Distribution (Vertical)
- Top 20 most common words
- Bar chart with counts
- Rotated labels for readability

#### 3.3 Frequency Distribution (Horizontal)
- Same data, horizontal layout
- Better for long word names
- Color: coral/orange

#### 3.4 Token Length Distribution
- Histogram + box plot
- Shows distribution of text lengths
- Identifies outliers

#### 3.5 Word Length Distribution
- Distribution of character counts
- Helps identify text characteristics

**Output Files**:
- `wordcloud.png`
- `frequency_distribution.png`
- `frequency_horizontal.png`
- `token_length_distribution.png`
- `word_length_distribution.png`

---

### 4. Model Selection Documentation

The `BERT_vs_VADER_Analysis.md` document provides:
- Detailed technical comparison
- Performance benchmarks
- Cost analysis
- Implementation recommendations
- Code examples
- Decision framework

**Key Insights**:
- **VADER**: Fast, lightweight, good for prototypes
- **BERT**: Accurate, handles nuance, requires more resources
- **Recommendation**: Start with VADER, migrate to BERT for production

---

## 📊 Expected Results

### Sample Statistics
After running the complete pipeline, you should see:

```
DATA ACQUISITION
✓ Successfully collected 500 records
✓ Data saved to raw_data.csv

PREPROCESSING
✓ Processing complete! 485 records processed
  Average tokens per record: 8.3

EDA SUMMARY
✓ Total unique words: 892
✓ Total word occurrences: 4,025
✓ Top word: "python" (127 occurrences)
```

### Visualizations
All PNG files will be saved in the current directory with:
- High resolution (300 DPI)
- Professional styling
- Clear labels and titles

---

## 🔧 Troubleshooting

### Common Issues

#### 1. NLTK Data Not Found
```bash
# Error: Resource punkt not found
# Solution:
python -c "import nltk; nltk.download('punkt')"
```

#### 2. snscrape Not Working
```bash
# Error: snscrape module not found
# Solution:
pip install snscrape --break-system-packages

# Note: snscrape may have Twitter API changes
# The script will automatically fall back to sample data
```

#### 3. WordCloud Generation Fails
```bash
# Error: No module named 'PIL'
# Solution:
pip install pillow --break-system-packages
```

#### 4. Empty Processed Data
- Check if raw_data.csv exists
- Verify keyword matches actual content
- Adjust preprocessing settings (may be too aggressive)

#### 5. Matplotlib Backend Issues
```bash
# Error: $DISPLAY not set
# Solution: Add to script
import matplotlib
matplotlib.use('Agg')
```

---

## 🎓 Assignment Completion Checklist

- [ ] **Task 1**: Data acquisition script fetches 500+ records
- [ ] **Task 2**: Preprocessing pipeline implemented with Regex/NLTK
- [ ] **Task 3**: Word cloud generated and saved
- [ ] **Task 3**: Frequency distribution of top 20 words created
- [ ] **Task 4**: BERT vs VADER comparison documented

---

## 📈 Next Steps (Future Assignments)

### Assignment 2: Model Implementation
- Implement VADER sentiment analyzer
- Fine-tune BERT model
- Compare performance metrics
- Create accuracy reports

### Assignment 3: Dashboard Development
- Build Flask/Streamlit web interface
- Real-time data streaming
- Interactive visualizations
- User authentication

### Assignment 4: Deployment
- Containerize with Docker
- Deploy to cloud (AWS/GCP/Azure)
- Set up monitoring
- API endpoints

---

## 🤝 Contributing

This is an educational project. Suggestions for improvement:
- Additional data sources (Reddit, Instagram)
- More advanced preprocessing techniques
- Alternative visualization libraries (Plotly, Bokeh)
- Sentiment trend analysis over time

---

## 📝 License

This project is created for educational purposes. Feel free to use and modify for your learning.

---

## 📚 Resources

### Documentation
- NLTK: https://www.nltk.org/
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/
- WordCloud: https://github.com/amueller/word_cloud

### Research Papers
- VADER: Hutto & Gilbert (2014)
- BERT: Devlin et al. (2019)

### Tutorials
- Twitter API: https://developer.twitter.com/en/docs
- Sentiment Analysis: https://realpython.com/sentiment-analysis-python/

---

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Consult the BERT vs VADER document
4. Experiment with different parameters

---

**Good luck with your sentiment analysis project! 🚀**
