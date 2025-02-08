# Sentiment Analysis App

A Streamlit-based text classification application for sentiment analysis using machine learning.

## Features

- Text sentiment analysis (positive/negative classification)
- Interactive web interface
- Real-time predictions
- Sample dataset for training
- Simple and intuitive UI

## Technologies Used

- Python 3.11
- Streamlit
- scikit-learn
- NLTK
- pandas

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd sentiment-analysis-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Launch the application
2. Click "Load Sample Data & Train Model" in the sidebar
3. Enter your text in the input area
4. Click "Analyze Sentiment" to get the prediction

## Project Structure

```
├── model/
│   ├── classifier.py      # Sentiment classification model
│   └── preprocessor.py    # Text preprocessing utilities
├── styles/
│   └── custom.css        # Custom styling
├── utils/
│   └── sample_data.py    # Sample dataset loader
└── main.py               # Main Streamlit application
```

## License

MIT License
