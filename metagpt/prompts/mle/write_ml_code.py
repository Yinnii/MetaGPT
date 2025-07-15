ML_SYSTEM_MESSAGE = """You are a highly skilled machine learning engineer. 
Your task is to write Python code that trains a machine learning model based on the provided specifications.
The code should be efficient, well-structured, and include necessary imports and comments for clarity."""

ML_PROMPT = """You are given a task to write Python code for a machine learning model. 
# User requirements
datasets_dir: "/home/yin/Projects/MetaGPT/metagpt/ext/sela/SELA_datasets/{{dataset_name}}" # path to the datasets directory

# Configuration
{configuration}

Please follow the specifications below:
1. Import necessary libraries such as defined in the configuration.
3. The data path is specified in the user requirements.
2. Use the model type specified in the configuration.
4. Use the hyperparameters specified in the configuration.
5. Train and test the model using the datasets in the datasets directory. The dataset is always name train.csv
6. Save the model as a pickle file in the working directory.

# Output
While some concise thoughts are helpful, code is absolutely required. 
Always output one and only one code block in your response. Output code in the following format:
```python
your code
```

# Example
Configuration:  
  ```json   {{
                "dataset": {{  
                  "dataset_name": "creditg",
                  "target_column": "class"
            }},
            "flow": {{
                "implementation": "sklearnensembleforestRandomForestClassifier2083190",
                "software": [
                    {{
                        "name": "scipy",
                        "version": "0.9"
                    }},
                    {{
                        "name": "numpy",
                        "version": "1.6.1"
                    }},
                    {{
                        "name": "sklearn",
                        "version": "0.18.1"
                    }}
                ],
                "hyperparametersettings": {{
                    "stepsSetting2083190": "[{{\"oml-python:serialized_object\": \"component_reference\", \"value\": {{\"key\": \"pca\", \"step_name\": \"pca\"}}}}, {{\"oml-python:serialized_object\": \"component_reference\", \"value\": {{\"key\": \"randomforestclassifier\", \"step_name\": \"randomforestclassifier\"}}}}]"
                }}
            }},
            "evaluation": {{
                "measure": "predictive_accuracy",
                "value": 0.784
            }}
          }}
  ```

Output:
```python
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
# Constants
DATASET_NAME = "creditg"
TARGET_COLUMN = "class"
DATASETS_DIR = "/home/yin/Projects/MetaGPT/metagpt/ext/sela/SELA_datasets" + DATASET_NAME 

def load_data(dataset_name):
    # Load the dataset from the specified directory
    data_path = os.path.join(DATASETS_DIR, f"/train.csv") # DO NOT ADJUST THE FILENAME
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    # Separate features and target variable
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    
    # Convert categorical columns to numeric
    X = pd.get_dummies(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_pca, y_train)
    
    return model, pca

def evaluate_model(model, pca, X_test, y_test):
    # Transform test data using PCA
    X_test_pca = pca.transform(X_test)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, pca):
    # Ensure the workspace directory exists
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(os.path.join(DATSET_DIR, 'random_forest_model.pkl'), 'wb') as f:
        pickle.dump((model, pca), f)
    print(f"Model saved to {{os.path.join(DATASET_DIR, 'random_forest_model.pkl')}}")

def main():
    # Load and preprocess data
    data = load_data(DATASET_NAME)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model, pca = train_model(X_train, y_train)
    
    # Evaluate the model
    training_accuracy = evaluate_model(model, pca, X_train, y_train)
    print(f"Training Accuracy: {{training_accuracy:.3f}}")
    accuracy = evaluate_model(model, pca, X_test, y_test)
    print(f"Model Accuracy: {{accuracy:.3f}}")
    
    # Save the model
    save_model(model, pca)

if __name__ == "__main__":
    main()
```
"""

REFLECT_PROMPT = """You are a machine learning engineer.
Your task is to reflect on the previously created code and improve it based on the feedback given.
The structure should stay the same, just improve the code based on the feedback. 
Make sure the dataset directory is correct and is in the following format:
/home/yin/Projects/MetaGPT/metagpt/ext/sela/SELA_datasets/{{dataset_name}}/raw/train.csv

# Feedback
{feedback}

# Output
Output the improved code in the following format:
```python
your code
```
"""