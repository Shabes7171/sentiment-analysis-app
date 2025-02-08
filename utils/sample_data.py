import pandas as pd

def load_sample_data():
    """Load sample dataset for demonstration."""
    sample_data = {
        'text': [
            "This product is amazing! I love it.",
            "Terrible experience, would not recommend.",
            "Great customer service and quick delivery.",
            "The quality is poor and it broke quickly.",
            "Excellent value for money, very satisfied.",
            "Disappointing product, waste of money.",
            "Outstanding performance and reliability.",
            "Not worth the price, avoid this product.",
            "Perfect solution for my needs, very happy.",
            "Worst purchase ever, complete garbage."
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative
    }
    return pd.DataFrame(sample_data)
