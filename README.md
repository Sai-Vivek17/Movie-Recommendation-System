# 🎬 Content-Based Movie Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Executive Summary
A robust, content-based machine learning recommendation system that suggests movies to users based on their historical preferences. By leveraging Natural Language Processing (NLP) techniques and mathematical similarity metrics, this engine processes movie metadata (genres, keywords, cast, and director) to deliver highly accurate and personalized recommendations.

This project demonstrates proficiency in **data preprocessing, text vectorization, matrix operations, and algorithm optimization.**

## 🛠️ Technical Architecture & Tech Stack

* **Core Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
* **Fuzzy String Matching:** `difflib` (for robust user input handling)
* **Model Serialization:** `joblib` (for optimized storage and fast load times)

## 🧠 Methodology & Workflow

1. **Data Ingestion & Cleaning:** * Imported movie dataset containing thousands of records.
   * Handled missing values (NaN) by replacing them with empty strings to prevent data leakage and pipeline crashes.
   * Extracted and concatenated relevant text features: `genres`, `keywords`, `tagline`, `cast`, and `director`.

2. **Feature Engineering (NLP):**
   * Transformed the aggregated text data into numerical feature vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**.
   * This process weighted the significance of specific words across the dataset, filtering out common noise and highlighting unique movie identifiers.

3. **Similarity Computation:**
   * Calculated the **Cosine Similarity** matrix across all vectorized movie profiles.
   * Achieved an efficient pairwise comparison to determine the multidimensional angle (similarity score) between any two given movies.

4. **Recommendation Logic & Error Handling:**
   * Implemented a fuzzy matching algorithm using `difflib` to map user input (even with typos) to the closest valid movie title in the database.
   * Retrieved the similarity array for the target movie, sorted the scores in descending order, and returned the Top $N$ closest matches.
   * Modularized the architecture by serializing the similarity matrix (`.pkl`), decoupling the training phase from the inference phase for deployment readiness.
  
Since the size of the models exceeds 100mb, An accesable google drive link of the models is attached[(https://drive.google.com/drive/folders/1QCRg_ZDvffu59tsbsSSfw5YA_i47LRdb?usp=sharing)]

## 📂 Repository Structure
```text
├── data/                # Folder for the dataset
    └── movies dataset.csv
├── models/                 # Saved machine learning models (.pkl)
    └── movies_data.pkl
    └── similarity_matrix.pkl
├── notebooks/              # Jupyter/Colab notebooks for exploration and training
│   └── Movie_Prediction_system.ipynb
├── requirements.txt        # List of dependencies required to run the code
└── README.md               # Project documentation

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
