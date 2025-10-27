import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init for KMeans
cluster_labels = kmeans.fit_predict(X_scaled)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Set page title
st.title('Breast Cancer Prediction App')

# Create input form for features
st.sidebar.header('Input Features')

def user_input_features():
    features = {}

    # Create input fields for all features
    for feature in data.feature_names: # Use data.feature_names directly
        features[feature] = st.sidebar.slider(
            feature,
            float(X[feature].min()),
            float(X[feature].max()),
            float(X[feature].mean())
        )

    return pd.DataFrame(features, index=[0])

# Get user input
user_input = user_input_features()

# Display the input features
st.subheader('User Input Features')
st.write(user_input)

# Prepare the input for prediction
# We need to scale the user input using the same scaler fitted on the training data
user_input_scaled = scaler.transform(user_input)


# Make prediction
prediction = knn.predict(user_input_scaled)
prediction_proba = knn.predict_proba(user_input_scaled)

# Get clustering result
cluster = kmeans.predict(user_input_scaled)

# Display results
st.subheader('Prediction')
prediction_label = 'Benign' if prediction[0] == 1 else 'Malignant'
st.write(f'The tumor is predicted to be: **{prediction_label}**')

st.subheader('Prediction Probability')
st.write(f'Probability of being Malignant: {prediction_proba[0][0]:.2f}')
st.write(f'Probability of being Benign: {prediction_proba[0][1]:.2f}')

st.subheader('Cluster Assignment')
st.write(f'The sample belongs to cluster: {cluster[0]}')

# Optional: Add visualization of the clusters
st.subheader('Clustering Visualization')
fig, ax = plt.subplots()
# For visualization, it's better to use a dimensionality reduction technique like PCA
# to plot in 2D. For simplicity here, we'll just plot the first two scaled features.
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
ax.scatter(user_input_scaled[0, 0], user_input_scaled[0, 1], color='red', marker='*', s=200, label='User Input')
ax.set_xlabel(data.feature_names[0]) # Use actual feature names
ax.set_ylabel(data.feature_names[1]) # Use actual feature names
ax.set_title('Data Clusters with User Input') # Added title
ax.legend()
plt.colorbar(scatter, label='Cluster Label') # Added colorbar label
st.pyplot(fig)