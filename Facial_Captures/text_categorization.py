import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset from the CSV file
csv_file_path = "movie_reviews.csv"
df = pd.read_csv(csv_file_path)

# Ensure each category has a minimum number of samples
df = df.groupby('Genre').filter(lambda x: len(x) >= 2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Review Text'], df['Genre'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Support Vector Machine classifier using cross-validation
classifier = SVC()
cross_val_scores = cross_val_score(classifier, X_train_vectorized, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cross_val_scores.mean():.2f}")

# Train the model on the entire training set
classifier.fit(X_train_vectorized, y_train)

# Vectorize the test data
X_test_vectorized = vectorizer.transform(X_test)

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions, zero_division=1))  # Set zero_division to 1 to avoid warnings
