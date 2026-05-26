import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score # <-- NEW: Our grading tools

# 1. Load the dataset
df = pd.read_csv("data/housing.csv")

# 2. Select features and target
X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

# 3. Split the data (80% Practice Test, 20% Final Exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and train the model (The Student Studies)
model = LinearRegression()
print("Training the model...")
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. NEW: Grade the Final Exam (Evaluate the Model)
# Ask the model to guess the prices for the 20% test data
test_predictions = model.predict(X_test)

# Compare the guesses (test_predictions) to the real answers (y_test)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print("\n--- Model Report Card ---")
print(f"Average Error (MAE): The model's guesses are off by about ${mae:,.2f} on average.")
print(f"Accuracy Score (R-squared): {r2 * 100:.2f}%")
print("-------------------------\n")
# ---------------------------------------------------------

# 6. Predict the price for a brand new, hypothetical house
new_house = pd.DataFrame([[2000, 3, 2]], columns=["Area", "Bedrooms", "Bathrooms"])
prediction = model.predict(new_house)

print(f"Predicted Price for a 2000 sqft, 3 Bed, 2 Bath house: ${prediction[0]:,.2f}")