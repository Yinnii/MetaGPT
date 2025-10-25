DATASET_DESCRIPTION_SELA_PROMPT = """
# Dataset Description
{dataset}

# Dataset Metadata
{metadata}

# Dataset Head
{head}
"""

DATASET_DESCRIPTION_CUSTOM_PROMPT = """
# Dataset Description
{dataset_description}
"""
# konkretisiere mehr die prompts

INFORMATION_BASED_INSIGHTS = """
# Dataset Description
{description}

# Existing Insights
{predefined_insights}

You are a Machine Learning Expert tasked with analyzing a dataset and proposing insights to improve model performance.
The insights for each task type should contain 4 unique insights.
Be specific about models' choices, ensemble, tuning, preprocessing, and feature engineering techniques.
Your model choices should be advanced enough to be helpful and distinct from the predefined insights.
All insights should be distinct from each other.
The hyperparameter optimization insights should not use grid search or best estimators, use a random setting instead.

# Format
```json
[
    {{
        "task_type": "EDA",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
        ]   
    }},
    {{
        "task_type": "Data Preprocessing",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
        ]   
    }},
    {{
        "task_type": "Feature Engineering",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
        ]   
    }},
    {{
        "task_type": "Model Training",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
        ]   
    }},
    {{
        "task_type": "Hyperparameter Optimization",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
        ]
    }}
]

"""

DATASET_INSIGHT_PROMPT_ONTORAG = """
# Role
You are a Machine Learning Expert tasked with analyzing a dataset and proposing insights to improve model performance.
You will be provided with a dataset description and examples of hyperparameter settings from previous runs. 
First, use this information to generate insights for different task types in the machine learning pipeline.

# Dataset Description
{description}

# Information
{hyperparameter_settings_examples}

# Instruction
The insights should be proposed based on the dataset description and the information with different task types. 
The first three insights of each task type should be based on the Information provided, while the last two insights should be based on your own knowledge and experience.
Each task type should have at least 5 insights.
Be specific about models' choices, ensemble and tuning techniques, and preprocessing & feature engineering techniques.
Your model choices should be advanced enough to be helpful.

# Format
```json
[
    {{
        "task_type": "EDA",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Data Preprocessing",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Feature Engineering",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Model Training",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]   
    }},
    {{
        "task_type": "Hyperparameter Tuning",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
            ...
            "insightN"
        ]
    }}
]
```
"""

INSIGHT_PROPOSAL_PROMPT_ONTORAG = """
You are an AI assistant tasked with analyzing a machine learning solution and proposing new insights to improve its performance. 
Given the current solution code and development score, suggest innovative approaches to enhance the model.

Current Solution Code:
{solution_code}

Development Score: {dev_score}

Based on this information, propose 3-5 new insights across different aspects of the machine learning pipeline (Data Preprocessing, Feature Engineering, Model Training and Hyperparameter Optimization). 
Your insights should be specific, actionable, and have the potential to improve the model's performance.

Please format your response as a JSON array with the following structure:
[

    {{
        "task_type": "Data Preprocessing",
        "insights": [
            "insight1",
            "insight2"
        ]
    }},
    {{
        "task_type": "Feature Engineering",
        "insights": [
            "insight1",
            "insight2"
        ]
    }},
    {{
        "task_type": "Model Training",
        "insights": [
            "insight1",
            "insight2"
        ]
    }},
    {{
        "task_type": "Hyperparameter Optimization",
        "insights": [
            "insight1",
            "insight2"
        ]
    }}
]
"""

MALEX_RUN_PROMPT = """ Write and run a python script to train the dataset with the following configuration: 
{configuration}
"""

INSIGHT_PROPOSER = """ You are a Prompt Engineer. 
Given the following software {software} and classifier {classifier}, provide a concise insight prompt for training a model with it. 
Formulate the prompt in a way that it can be directly used to guide the model training process. 
Remove special characters and ensure clarity and precision in the prompt.
"""

INSIGHT_PROPOSER_EXAMPLE = """
Example:
Given the following software scikit-learn and classifier RandomForestClassifier, provide a concise insight prompt for training a model with it.
Output:
"Train a RandomForestClassifier using scikit-learn on the provided dataset. Use appropriate preprocessing, set relevant hyperparameters (e.g., n_estimators, max_depth), fit the model on the training data, and evaluate performance using cross-validation or a separate validation set."
"""

HYPERPARAMETER_PROPOSER = """ You are a Prompt Engineer. 
Given the following hyperparameter settings {hyperparameter_settings}, provide a concise insight prompt for optimizing hyperparameters with it. 
If the settings are empty, select random parameters. 
Formulate the prompt in a way that it can be used by different classification models by adapting parameter names as needed.
Remove special characters and ensure clarity and precision in the prompt.
"""

HYPERPARAMETER_PROPOSER_EXAMPLE = """
Example:
Given the following hyperparameter settings {'n_estimators': [100, 200], 'max_depth': [10, 20]}, provide a concise insight prompt for optimizing hyperparameters with it.
Output: 
"Optimize hyperparameters using n estimators set to 100 and 200 and max depth set to 10 and 20 to find the best model performance. Ensure compatibility with other classification models by adapting parameter names as needed."
"""