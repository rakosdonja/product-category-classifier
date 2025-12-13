# Product Category Classifier

## Project Overview
This project implements a machine learning system that automatically predicts the **product category** based solely on the **product title**.

It simulates a real-world e-commerce scenario where thousands of new products are added daily and must be categorized quickly, consistently, and accurately.

The project follows a **complete machine learning lifecycle**:
- data exploration and cleaning,
- feature engineering,
- model training and evaluation,
- saving a production-ready model,
- interactive prediction via command line.

---

## Business Problem
Manual product categorization is:
- time-consuming,
- error-prone,
- inconsistent across large teams.

This model automates the process by suggesting a category immediately after a product title is entered, improving:
- operational efficiency,
- data consistency,
- user experience on the platform.

---

## Dataset
The dataset (`products.csv`) contains approximately **35,000 products** with the following fields:

- `Product ID`
- `Product Title`
- `Merchant ID`
- `Category Label` (target variable)
- `Product Code`
- `Number of Views`
- `Merchant Rating`
- `Listing Date`

Only the **product title** and engineered text-based features are used for modeling.

---

## Exploratory Data Analysis (EDA)
EDA was performed in:

`notebooks/01_product_category_eda.ipynb`

It included:
- inspection of dataset structure and missing values,
- cleaning and standardization of category labels,
- removal of incomplete rows,
- class distribution analysis,
- creation and analysis of text-based features.

After standardization, the final dataset contains **10 well-balanced product categories**.

---

## Feature Engineering
From the raw product title, the following features were engineered:

- `product_title` – original text input,
- `title_length` – number of characters,
- `title_word_count` – number of words,
- `title_digit_count` – number of digits (model numbers, capacities, sizes),
- `title_special_char_count` – number of special characters.

These features help the model distinguish between categories such as CPUs, TVs, washing machines, and mobile phones.

---

## Modeling & Evaluation
Modeling was performed in:

`notebooks/02_product_category_modeling.ipynb`

### Models Tested
- Multinomial Naive Bayes  
- Logistic Regression  
- Linear Support Vector Classifier (Linear SVC)

### Final Model Choice
**Linear Support Vector Classifier (Linear SVC)** was selected as the final model due to:
- the highest overall accuracy,
- strong and consistent F1 scores across all categories,
- excellent performance on large, sparse TF-IDF feature spaces.

### Performance (Sanity-check 80/20 split)
- **Accuracy:** approximately **96–97%**
- High precision and recall across all major product categories

---

## Training Script (Production-Ready)
The final model is trained using a standalone, reproducible Python script:

`train_model.py`

This script:
- loads the dataset from `data/products.csv`,
- cleans and standardizes category labels,
- builds a preprocessing and modeling pipeline,
- performs a sanity-check evaluation,
- trains the final model on the full dataset,
- saves the trained pipeline to disk.

Run training with:

```bash
python train_model.py
```

The trained model is saved as:

```bash
model/product_category_model.pkl
```
Model artifacts are excluded from version control via .gitignore.

---

## Interactive Prediction

The project includes an interactive command-line tool for testing the model:

Run:

```bach
python predict_category.py
```
Example:
Enter product title: iphone 7 32gb gold
Predicted category: mobile phones
This simulates a real-world workflow where a new product title is entered into the system and an appropriate category is suggested instantly.

---

## Project Structure

The repository is organized as follows:

product-category-classifier/

│

├── data/

│   └── products.csv

│

├── notebooks/

│   ├── 01_product_category_eda.ipynb

│   └── 02_product_category_modeling.ipynb

│

├── model/

│   └── product_category_model.pkl   (ignored by git)

│

├── train_model.py

├── predict_category.py

├── feature_utils.py

├── .gitignore

└── README.md

This structure ensures clarity, reproducibility, and easy collaboration in a team environment.

---

## Technologies Used

Python

pandas, numpy

scikit-learn

TF-IDF Vectorization

Git & GitHub

Jupyter Notebook / Google Colab

---

## Key Takeaways

Careful text preprocessing and feature engineering significantly improve classification performance.

Linear models perform exceptionally well on large, sparse text feature spaces.

Clean project structure and reproducible scripts are essential for production-ready machine learning solutions.
