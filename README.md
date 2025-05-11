# 🛡️ Cyberbullying Detection using SVM and NLTK

A machine learning project for detecting cyberbullying in text messages using **Support Vector Machines (SVM)** and **Natural Language Toolkit (NLTK)**. This project leverages text preprocessing techniques, feature extraction, and SVM classification to identify abusive, harmful, or offensive content, helping to create a safer digital environment.

## 📂 Key Features

* Text preprocessing with NLTK (tokenization, stopword removal, stemming, and lemmatization)
* TF-IDF vectorization for feature extraction
* SVM for robust text classification
* Model evaluation with precision, recall, and F1-score metrics
* Ready-to-use Jupyter notebook for quick experimentation
* AI-based cross-verification with Gemini API (optional)
* Image-to-text analysis for detecting harmful content in images

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Jupyter Notebook (recommended)
* Required libraries: `nltk`, `scikit-learn`, `pandas`, `numpy`, `openai`

### Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/your-username/cyberbullying-detection-svm-nltk.git
cd cyberbullying-detection-svm-nltk
pip install -r requirements.txt
```

### Dataset

Make sure to include a labeled dataset for training and testing. You can use publicly available datasets or create your own. Add it to the **Dataset.json** file.

### Project Structure

```
.
├── Dataset.json                # Labeled dataset for training and testing
├── cyberbully_model.pkl        # Trained SVM model
├── vectorizer.pkl              # TF-IDF vectorizer
├── train.py                    # Script to train the SVM model
├── test_model.py               # Script to evaluate the trained model
├── predict.py                  # Script for single text predictions
├── predictWithAi.py            # AI-based cross-verification using Gemini API
├── gemini_ai.py                # Helper script for Gemini AI integration
├── imgToText.py                # Extract text from images for analysis
├── requirements.txt            # List of required Python packages
├── README.md                   # Project documentation
└── LICENSE                     # Project license
```

### Running the Project

1. Run the Jupyter notebook:

```bash
jupyter notebook Cyberbullying_Detection.ipynb
```

2. Use **train.py** to train the model:

```bash
python train.py
```

3. Test the model:

```bash
python test_model.py
```

4. Make single predictions:

```bash
python predict.py
```

5. Use **predictWithAi.py** for enhanced accuracy with AI:

```bash
python predictWithAi.py
```

## 📊 Model Evaluation

Includes metrics like precision, recall, F1-score, and confusion matrix for performance analysis.

## 🔗 Future Work

* Integration with real-time social media data
* Improved accuracy using deep learning models
* API for seamless integration into web apps
* Enhanced sentiment analysis for better classification

## 🤝 Contributing

Feel free to open issues, share ideas, or submit pull requests to improve this project!

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## ⭐ Show Your Support

If you found this project helpful, please give it a star ⭐ and share it with others!

## 📧 Contact

For any questions or collaboration, reach out to me via email at kameshk0011@gmail.com.
