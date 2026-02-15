# Quick Start Guide - Sentiment Analysis Assignment 1

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
pip install pandas nltk matplotlib seaborn wordcloud vaderSentiment --break-system-packages
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Run Complete Pipeline
```bash
python run_complete_pipeline.py
```

**OR** run step-by-step:
```bash
python data_acquisition.py      # Get 500+ records
python preprocessing.py          # Clean text
python eda_analysis.py          # Create visualizations
```

---

## 📂 What You Get

### Scripts (Python Files)
- **data_acquisition.py** - Fetches Twitter data or generates samples
- **preprocessing.py** - Cleans text with Regex & NLTK
- **eda_analysis.py** - Creates word clouds & frequency plots
- **run_complete_pipeline.py** - Runs everything automatically
- **vader_vs_bert_demo.py** - Interactive comparison demo

### Documentation
- **BERT_vs_VADER_Analysis.md** - Complete model comparison (Task 4)
- **README.md** - Full project documentation
- **requirements.txt** - All dependencies

### Outputs (after running)
- **raw_data.csv** - Original 500+ records
- **processed_data.csv** - Cleaned text + tokens
- **wordcloud.png** - Word cloud visualization
- **frequency_distribution.png** - Top 20 words bar chart
- **frequency_horizontal.png** - Horizontal view
- **token_length_distribution.png** - Length analysis
- **word_length_distribution.png** - Word size analysis

---

## ✅ Assignment 1 Tasks

| Task | File | Status |
|------|------|--------|
| 1. Fetch 500+ records | `data_acquisition.py` | ✓ Ready |
| 2. Preprocessing pipeline | `preprocessing.py` | ✓ Ready |
| 3a. Word cloud | `eda_analysis.py` | ✓ Ready |
| 3b. Top 20 frequency plot | `eda_analysis.py` | ✓ Ready |
| 4. BERT vs VADER | `BERT_vs_VADER_Analysis.md` | ✓ Ready |

---

## 🎯 Customization

### Change Search Keyword
Edit in `data_acquisition.py`:
```python
KEYWORD = "Python programming"  # Change this
NUM_RECORDS = 500               # Or this
```

### Adjust Preprocessing
Edit in `preprocessing.py`:
```python
preprocessor = TextPreprocessor(
    use_lemmatization=True,     # False for stemming
    remove_stopwords=True       # False to keep stopwords
)
```

### Customize Visualizations
Edit in `eda_analysis.py`:
```python
eda.generate_wordcloud(
    max_words=100,              # More/fewer words
    colormap='viridis'          # Different colors
)
```

---

## 🐛 Common Issues

**Error: NLTK data not found**
```bash
python -c "import nltk; nltk.download('all')"
```

**Error: No module named 'wordcloud'**
```bash
pip install wordcloud pillow --break-system-packages
```

**Error: processed_data.csv not found**
→ Run scripts in order: data_acquisition → preprocessing → eda

**Want real Twitter data?**
→ Set environment variable: `export TWITTER_BEARER_TOKEN='your_token'`
→ Otherwise, script uses sample data (which works fine!)

---

## 🚀 Next Steps

1. Run the pipeline: `python run_complete_pipeline.py`
2. Review visualizations (all .png files)
3. Read `BERT_vs_VADER_Analysis.md` for model insights
4. Try the demo: `python vader_vs_bert_demo.py`
5. Customize and experiment!

---

## 📊 Expected Runtime

- Data acquisition: 30-60 seconds
- Preprocessing: 10-20 seconds  
- EDA: 15-30 seconds
- **Total: ~2-3 minutes**

---

## 🎓 For Your Report

Include these outputs:
1. Sample of raw vs cleaned text (from `processed_data.csv`)
2. Word cloud image (`wordcloud.png`)
3. Frequency distribution (`frequency_distribution.png`)
4. Key insights from BERT vs VADER comparison
5. Summary statistics (printed by scripts)

---

## 💡 Tips

- **No Twitter API?** No problem! Sample data works great for learning
- **Slow performance?** Reduce `NUM_RECORDS` to 200-300
- **Want more words in cloud?** Edit `max_words` parameter
- **Different colors?** Try colormaps: 'plasma', 'inferno', 'magma', 'cividis'

---

## 📞 Help

1. Check `README.md` for detailed docs
2. Read error messages carefully
3. Verify files exist: `ls -l *.csv *.png`
4. Run scripts individually to isolate issues

**All files are ready to use - just run and customize! 🎉**
