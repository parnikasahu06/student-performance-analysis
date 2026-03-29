import matplotlib.pyplot as plt
import pandas as pd 

df=pd.read_csv("student_data.csv")

df=df.loc[:, ~df.columns.str.contains("Unnamed")]

print(df.head())
print(df.columns)

print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())

print(df.corr())

top_students = df.sort_values(by="Marks", ascending=False).head(5)
print(top_students)

# Graph 1: Study Hours vs Marks
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

# Graph 2: Social Media vs Marks
plt.scatter(df["Social_Media_Hours"], df["Marks"])
plt.xlabel("Social Media Hours")
plt.ylabel("Marks")
plt.title("Social Media vs Marks")
plt.show()

# Graph 3: Attendence vs Marks
plt.scatter(df["Attendence"], df["Marks"])
plt.xlabel("Attendence")
plt.ylabel("Marks")
plt.title("Attendence vs Marks")
plt.show()

print("\n--- Insights ---")
print("1. Study hours strongly increase marks.")
print("2. Attendence has a positive impact on performance.")
print("3. Social media usage negatively affects marks.")
print("4. Sleep has less impact compared to other factors.")