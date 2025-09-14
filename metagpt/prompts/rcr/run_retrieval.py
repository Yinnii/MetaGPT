RUN_RETRIEVAL_PROMPT = """ 
  Given a code cell and its outputs, extract the relevant information about the model used, its hyperparameters, and its evaluation metrics.
  Use a random ID for the run name. 
"""

RUN_RETRIEVAL_EXAMPLE = """
  Here is an example extract of a single code cell and its outputs:
  ...
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6c97016b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-08-26T14:32:41.026493Z",
     "iopub.status.busy": "2025-08-26T14:32:41.025876Z",
     "iopub.status.idle": "2025-08-26T14:33:36.630431Z",
     "shell.execute_reply": "2025-08-26T14:33:36.629804Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 Score on Dev Set: 0.9103\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions saved to dev_predictions.csv and test_predictions.csv.\n"
     ]
    }
   ],
   "source": [
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import f1_score\n",
    "\n",
    "# Prepare the data for training and evaluation\n",
    "X_train = train_features.drop(columns=['Class'])\n",
    "y_train = train_features['Class']\n",
    "X_dev = dev_features.drop(columns=['Class'])\n",
    "y_dev = dev_features['Class']\n",
    "\n",
    "# Check for unique classes in the target variable\n",
    "unique_classes = y_train.unique()\n",
    "num_classes = len(unique_classes)\n",
    "\n",
    "# Initialize the model with the correct number of classes\n",
    "model = XGBClassifier(\n",
    "    objective='multi:softmax',  # Assuming 'Class' is a multi-class target\n",
    "    eval_metric='mlogloss',\n",
    "    use_label_encoder=False,\n",
    "    n_estimators=100,\n",
    "    learning_rate=0.1,\n",
    "    max_depth=6,\n",
    "    num_class=num_classes,  # Set the number of classes\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions on the dev set\n",
    "y_dev_pred = model.predict(X_dev)\n",
    "\n",
    "# Calculate F1 score\n",
    "f1 = f1_score(y_dev, y_dev_pred, average='weighted')\n",
    "\n",
    "# Print the F1 score\n",
    "print(f\"F1 Score on Dev Set: {f1:.4f}\")\n",
    "\n",
    "# Prepare test predictions\n",
    "X_test = test_features\n",
    "y_test_pred = model.predict(X_test)\n",
    "\n",
    "# Save predictions to CSV files\n",
    "dev_predictions = pd.DataFrame({'target': y_dev_pred})\n",
    "test_predictions = pd.DataFrame({'target': y_test_pred})\n",
    "\n",
    "dev_predictions.to_csv('../../../workspace/icr/dev_predictions.csv', index=False)\n",
    "test_predictions.to_csv('../../../workspace/icr/test_predictions.csv', index=False)\n",
    "\n",
    "print(\"Predictions saved to dev_predictions.csv and test_predictions.csv.\")\n"
   ]
  }
  ...

  Based on the example extract, the following information can be retrieved:
  {
    "name": "run34567", <- run with a random ID assigned
    "flow": {
      "implementation": "XGBClassifier",
      "software": "sklearn",
      "hyperparametersettings": {
          "objective": 'multi:softmax',
          "eval_metric": 'mlogloss',
          "use_label_encoder": False,
          "learning_rate": 0.1,
          "max_depth": 6,
          "num_class": len(unique_classes),
          "random_state": 42
      }
    },
    "evaluation": {
      "measure": "predictive_accuracy",
      "value": 0.9103
    }
  }

  This is only an example, the actual implementation may vary.
"""