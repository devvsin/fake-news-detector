# 📰 Fake News Detector

An AI-powered web application that uses machine learning to detect and classify fake news articles in real-time. Built with Python, Streamlit, and scikit-learn.



## 🚀 Features

- **Real-time Analysis**: Instant fake news detection with confidence scores
- **Interactive Web Interface**: Clean, user-friendly Streamlit application
- **Text Analysis**: Detailed breakdown of linguistic patterns and indicators
- **Example Articles**: Pre-loaded examples to test the model
- **File Upload Support**: Analyze text files directly
- **Confidence Metrics**: Visual probability breakdowns and confidence levels
- **Debugging Tools**: See exactly how the AI makes its decisions


## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/devvsin/fake-news-detector.git
cd fake-news-detector
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
**⚠️ Important: The dataset is not included in this repository due to size limitations.**

You'll need to download a fake news dataset.
- **[Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** 

**Setup Instructions:**
1. Create a `data/` folder in your project directory
2. Download your chosen dataset 
3. Place CSV files as:
   - `data/Fake.csv` - Fake news articles (must have 'title' and 'text' columns)
   - `data/True.csv` - Real news articles (must have 'title' and 'text' columns)

**Quick Test with Sample Data:**
If you want to test without downloading large datasets, you can use the provided sample files in `sample_data/` to ensure everything works.

### Step 5: Train the Model
```bash
python train_model.py
```

This will create:
- `model/model.pkl` - Trained logistic regression model
- `model/vectorizer.pkl` - TF-IDF vectorizer
- `model/evaluation.txt` - Performance metrics

### Step 6: Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎮 Usage

### Basic Usage
1. **Launch the app**: Run `streamlit run app.py`
2. **Enter text**: Paste a news article in the text area
3. **Analyze**: Click "🔍 Analyze Article" button
4. **View results**: See prediction, confidence score, and detailed analysis

### Advanced Features
- **File Upload**: Upload `.txt` files containing news articles
- **Example Testing**: Use pre-loaded real and fake news examples
- **Text Analysis**: Expand the "Text Analysis Details" section to see:
  - Text statistics (word count, character count)
  - Preprocessed text preview
  - Linguistic indicators for fake/real news
  - Model confidence reasoning

### Interpreting Results
- **High Confidence (>80%)**: Strong prediction, reliable result
- **Moderate Confidence (60-80%)**: Good prediction, consider additional verification
- **Low Confidence (<60%)**: Uncertain prediction, manual verification recommended

## 📊 Dataset

The model is trained on a dataset containing:
- **Real News**: Legitimate news articles from reliable sources
- **Fake News**: Fabricated, misleading, or false news articles

### Data Preprocessing
- Text cleaning (remove URLs, numbers, punctuation)
- Lowercase conversion
- TF-IDF vectorization with 5,000 features
- Stop words removal

## 🤖 Model Details

### Architecture
- **Algorithm**: Logistic Regression
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 5,000
- **Preprocessing**: Text cleaning and normalization

### Performance Metrics
*[Performance metrics will be displayed here after training]*

### Training Process
1. Load and combine fake/real news datasets
2. Clean and preprocess text data
3. Split data (80% training, 20% testing)
4. Extract TF-IDF features
5. Train logistic regression model
6. Evaluate performance and save model

## 📁 Project Structure

```
fake-news-detector/
│
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
│
├── data/                 # Dataset folder (not in repository)
│   ├── Fake.csv         # Fake news articles
│   └── True.csv         # Real news articles
│
├── model/               # Trained models (not in repository)
│   ├── model.pkl       # Trained classifier
│   ├── vectorizer.pkl  # TF-IDF vectorizer
│   └── evaluation.txt  # Model performance metrics
│
└── screenshots/         # App screenshots for README
    ├── main_interface.png
    ├── results.png
    └── analysis.png
```

## 📦 Requirements

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=1.5.0
joblib>=1.3.0
numpy>=1.24.0
```

## 🧠 How It Works

1. **Text Input**: User provides a news article text
2. **Preprocessing**: Text is cleaned and normalized
3. **Feature Extraction**: TF-IDF converts text to numerical features
4. **Classification**: Logistic regression model predicts fake/real
5. **Analysis**: App provides confidence scores and linguistic analysis
6. **Results**: User sees prediction with detailed explanations

## 🔍 Key Features Explained

### Text Analysis Indicators

**Fake News Indicators:**
- Sensationalist language ("BREAKING", "SHOCKING")
- Conspiracy terms ("cover-up", "they don't want you to know")
- Anonymous sources ("insider", "secret source")
- Excessive punctuation and emotional language

**Real News Indicators:**
- Expert sources ("Dr.", "Professor", "Researcher")
- Institutional backing ("University", "Institute")
- Data and statistics
- Proper attribution and citations

## 🚀 Future Enhancements

- [ ] Deep learning models (BERT, LSTM)
- [ ] Multi-language support
- [ ] Source credibility analysis
- [ ] Fact-checking integration
- [ ] Browser extension
- [ ] API endpoint for integration
- [ ] Batch processing for multiple articles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Steps to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. While the model shows good performance, it should not be the sole method for determining news authenticity. Always verify information from multiple reliable sources.

## 🙏 Acknowledgments

- Dataset providers and the open-source community
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools
- Contributors and testers



⭐ **If you found this project helpful, please give it a star!** ⭐
