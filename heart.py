import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('heart_disease.csv')



# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function to predict heart disease
def predict_heart_disease():
    # Get user input
    age = float(entry_age.get())
    sex = float(entry_sex.get())
    cp = float(entry_cp.get())
    trestbps = float(entry_trestbps.get())
    chol = float(entry_chol.get())
    fbs = float(entry_fbs.get())
    restecg = float(entry_restecg.get())
    thalach = float(entry_thalach.get())
    exang = float(entry_exang.get())
    oldpeak = float(entry_oldpeak.get())
    slope = float(entry_slope.get())
    ca = float(entry_ca.get())
    thal = float(entry_thal.get())

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
        'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
        'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
    })

    # Standardize the user input
    user_data = scaler.transform(user_data)

    # Make a prediction
    prediction = model.predict(user_data)

    # Display the prediction
    if prediction[0] == 0:
        messagebox.showinfo("Prediction", "No Heart Disease")
    else:
        messagebox.showinfo("Prediction", "Heart Disease Detected")

# Create the main window
root = tk.Tk()
root.title("Heart Disease Prediction")

# Create input fields
tk.Label(root, text="Age").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sex (0: Female, 1: Male)").grid(row=1, column=0)
entry_sex = tk.Entry(root)
entry_sex.grid(row=1, column=1)

tk.Label(root, text="Chest Pain Type (0-3)").grid(row=2, column=0)
entry_cp = tk.Entry(root)
entry_cp.grid(row=2, column=1)

tk.Label(root, text="Resting Blood Pressure").grid(row=3, column=0)
entry_trestbps = tk.Entry(root)
entry_trestbps.grid(row=3, column=1)

tk.Label(root, text="Serum Cholesterol").grid(row=4, column=0)
entry_chol = tk.Entry(root)
entry_chol.grid(row=4, column=1)

tk.Label(root, text="Fasting Blood Sugar (0: <=120, 1: >120)").grid(row=5, column=0)
entry_fbs = tk.Entry(root)
entry_fbs.grid(row=5, column=1)

tk.Label(root, text="Resting Electrocardiographic Results (0-2)").grid(row=6, column=0)
entry_restecg = tk.Entry(root)
entry_restecg.grid(row=6, column=1)

tk.Label(root, text="Maximum Heart Rate Achieved").grid(row=7, column=0)
entry_thalach = tk.Entry(root)
entry_thalach.grid(row=7, column=1)

tk.Label(root, text="Exercise Induced Angina (0: No, 1: Yes)").grid(row=8, column=0)
entry_exang = tk.Entry(root)
entry_exang.grid(row=8, column=1)

tk.Label(root, text="ST Depression Induced by Exercise").grid(row=9, column=0)
entry_oldpeak = tk.Entry(root)
entry_oldpeak.grid(row=9, column=1)

tk.Label(root, text="Slope of the Peak Exercise ST Segment (0-2)").grid(row=10, column=0)
entry_slope = tk.Entry(root)
entry_slope.grid(row=10, column=1)

tk.Label(root, text="Number of Major Vessels (0-3)").grid(row=11, column=0)
entry_ca = tk.Entry(root)
entry_ca.grid(row=11, column=1)

tk.Label(root, text="Thalassemia (0-3)").grid(row=12, column=0)
entry_thal = tk.Entry(root)
entry_thal.grid(row=12, column=1)

# Create a prediction button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease)
predict_button.grid(row=13, column=0, columnspan=2)

# Run the main loop
root.mainloop()
