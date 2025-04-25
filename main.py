import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the Id column as it's not needed
df = df.drop('Id', axis=1)

# Create visualizations
plt.figure(figsize=(15, 10))

# Scatter plots for different feature combinations
plt.subplot(2, 2, 1)
for species in df['Species'].unique():
    species_data = df[df['Species'] == species]
    plt.scatter(species_data['SepalLengthCm'], species_data['SepalWidthCm'], label=species)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)') 
plt.title('Sepal Length vs Width by Species')
plt.legend()

# Box plots
plt.subplot(2, 2, 2)
df.boxplot(column=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], by='Species')
plt.title('Feature Distributions by Species')
plt.suptitle('')  # Remove automatic suptitle

# Histogram
plt.subplot(2, 2, 3)
for feature in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    plt.hist(df[feature], alpha=0.5, label=feature, bins=20)
plt.title('Distribution of Features')
plt.legend()

# Scatter plot of petal features
plt.subplot(2, 2, 4)
for species in df['Species'].unique():
    species_data = df[df['Species'] == species]
    plt.scatter(species_data['PetalLengthCm'], species_data['PetalWidthCm'], label=species)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Width by Species')
plt.legend()

plt.tight_layout()
plt.savefig('iris_visualizations.png')
plt.close()

# Prepare data for machine learning
X = df.drop('Species', axis=1)
y = df['Species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'iris_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')

print("\nModel and scaler have been saved as 'iris_model.pkl' and 'iris_scaler.pkl'")