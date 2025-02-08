git clone https://github.com/Shabes7171/sentiment-analysis-app.git
cd sentiment-analysis-app
```

2. Install the required packages:
```bash
pip install streamlit scikit-learn nltk pandas
```

3. Run the application:
```bash
streamlit run main.py
```

## Deployment

This app is deployed using [Streamlit Cloud](https://streamlit.io/cloud). To deploy your own version:

1. Fork this repository
2. Visit [Streamlit Cloud](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select this repository and the main file (main.py)
6. Click "Deploy"

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