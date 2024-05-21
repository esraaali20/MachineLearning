import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('FINAL.csv')

# Drop unnecessary columns
columns_to_drop = ['ID',  'Date', 'actual_price',  'Sales', 'Units_Sold','Total_Price']
df = df.drop(columns_to_drop, axis=1)
df = df.dropna()

# Separate features (X) and target (y) for category prediction
X_category = df['name']  # Features
y_category = df['category']  # Target

# Separate features (X) and target (y) for section prediction
X_section = df['name']  # Features
y_section = df['Category_Section']  # Target

# Splitting the data into training and testing sets for category prediction
X_train_category, X_test_category, y_train_category, y_test_category = train_test_split(X_category, y_category, test_size=0.2, random_state=42)

# Creating TF-IDF vectorizer for category prediction
tfidf_vectorizer_category = TfidfVectorizer()

# Fit and transform the training data for category prediction
X_train_tfidf_category = tfidf_vectorizer_category.fit_transform(X_train_category)

# Transform the test data for category prediction
X_test_tfidf_category = tfidf_vectorizer_category.transform(X_test_category)

# Initialize and train the DecisionTreeClassifier for category prediction
pre_category = DecisionTreeClassifier()
pre_category.fit(X_train_tfidf_category, y_train_category)

# Predicting on the test data for category prediction
predictions_category = pre_category.predict(X_test_tfidf_category)

# Calculating accuracy for category prediction
accuracy_category = accuracy_score(y_test_category, predictions_category)
print("Category Prediction Accuracy with DecisionTreeClassifier:", accuracy_category)

# Creating TF-IDF vectorizer for section prediction
tfidf_vectorizer_section = TfidfVectorizer()

# Fit and transform the training data for section prediction
X_train_tfidf_section = tfidf_vectorizer_section.fit_transform(X_section)

# Transform the test data for section prediction
X_test_tfidf_section = tfidf_vectorizer_section.transform(X_section)

# Initialize and train the DecisionTreeClassifier for section prediction
pre_section = DecisionTreeClassifier()
pre_section.fit(X_train_tfidf_section, y_section)

# Example of predicting the category and section for a new product description
new_product_description = ["This is a great Dress with amazing color"]  # Example new product description

# Convert the new product description to TF-IDF features for category prediction
new_product_tfidf_category = tfidf_vectorizer_category.transform(new_product_description)

# Use the trained DecisionTreeClassifier to predict the category
predicted_category = pre_category.predict(new_product_tfidf_category)
print("Predicted category:", predicted_category)

# Convert the new product description to TF-IDF features for section prediction
new_product_tfidf_section = tfidf_vectorizer_section.transform(new_product_description)

# Use the trained DecisionTreeClassifier to predict the section
predicted_section = pre_section.predict(new_product_tfidf_section)
print("Predicted section:", predicted_section)