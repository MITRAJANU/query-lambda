import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Dummy dataset
questions = [
    "What are the total sales for each product category?",
    "Provide a list of top 10 customers based on their purchase amounts.",
    "Show the average rating for each product.",
    "Which products have inventory levels below 100 units?",
    "What is the total revenue for the last quarter?"
]

# Corresponding Athena queries
athena_queries = [
    "SELECT product_category, SUM(sales) as total_sales FROM your_table GROUP BY product_category;",
    "SELECT customer_name, SUM(purchase_amount) as total_purchase FROM your_table GROUP BY customer_name ORDER BY total_purchase DESC LIMIT 10;",
    "SELECT product_name, AVG(rating) as average_rating FROM your_table GROUP BY product_name;",
    "SELECT product_name FROM your_table WHERE inventory < 100;",
    "SELECT SUM(revenue) as total_revenue FROM your_table WHERE date BETWEEN 'start_of_last_quarter' AND 'end_of_last_quarter';"
]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    questions, athena_queries, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)


def lambda_handler(event, context):
    # Access the "question" field directly from the Lambda event body
    question = event["body"]["question"]

    # Predict the Athena query using the trained model
    predicted_query = model.predict([question])[0]

    # Prepare and return the response
    response = {
        "statusCode": 200,
        "body": json.dumps({"predicted_query": predicted_query}),
    }

    return response
