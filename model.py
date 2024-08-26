import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the CSV file
df = pd.read_csv("churn-bigml.csv")

# Print the first few rows to understand the data
print(df.head())

# Encode categorical variables
label_encoders = {}
for column in ["International plan", "Voice mail plan", "Churn"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Select independent and dependent variables
X = df[["Account length", "Area code", "International plan", "Voice mail plan", 
        "Number vmail messages", "Total day minutes", "Total day calls", 
        "Total day charge", "Total eve minutes", "Total eve calls", 
        "Total eve charge", "Total night minutes", "Total night calls", 
        "Total night charge", "Total intl minutes", "Total intl calls", 
        "Total intl charge", "Customer service calls"]]

y = df["Churn"]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier(random_state=50)

# Fit the model
classifier.fit(X_train, y_train)

# Save the model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(sc, scaler_file)

# Save label encoders
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)


# Encode the target variable
label_encoder_y = LabelEncoder()
df['Churn'] = label_encoder_y.fit_transform(df['Churn'])

# Save the encoder for the target variable
with open('label_encoder_y.pkl', 'wb') as le_y_file:
    pickle.dump(label_encoder_y, le_y_file)