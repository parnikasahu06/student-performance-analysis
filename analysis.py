import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("student_data.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove unwanted unnamed columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Columns ---")
print(df.columns)

print("\n--- Statistics ---")
print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())

print("\n--- Correlation Matrix ---")
print(df.corr())

top_students = df.sort_values(by="Marks", ascending=False).head(5)

print("\n--- Top 5 Students ---")
print(top_students)

print("\n--- Insights ---")
print("1. Study hours strongly increase marks.")
print("2. Attendence has a positive impact on performance.")
print("3. Social media usage negatively affects marks.")
print("4. Sleep has less impact compared to other factors.")

# Graph 1: Study Hours vs Marks
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.grid(True)
plt.show()

# Graph 2: Social Media vs Marks
plt.scatter(df["Social_Media_Hours"], df["Marks"])
plt.xlabel("Social Media Hours")
plt.ylabel("Marks")
plt.title("Social Media vs Marks")
plt.grid(True)
plt.show()

# Graph 3: Attendance vs Marks
plt.scatter(df["Attendence"], df["Marks"])
plt.xlabel("Attendence")
plt.ylabel("Marks")
plt.title("Attendence vs Marks")
plt.grid(True)
plt.show()

print("\n--- Machine Learning Model ---")

# Features and target
X = df[["Study_Hours", "Attendence", "Sleep_Hours", "Social_Media_Hours"]]
y = df["Marks"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\n--- Model Predictions ---")
print("Predicted Marks:", y_pred)
print("Actual Marks:", list(y_test))

print("\n--- Model Performance ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\n--- Custom Prediction ---")
sample_data = [[7, 85, 6, 2]]
predicted_marks = model.predict(sample_data)

print("For a student with:")
print("Study Hours = 7")
print("Attendence = 85")
print("Sleep Hours = 6")
print("Social Media Hours = 2")
print("Predicted Marks =", predicted_marks[0])