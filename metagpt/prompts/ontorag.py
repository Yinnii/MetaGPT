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

# Existing tree
{predefined_insights}

You are a Machine Learning Expert tasked with analyzing a dataset and propsing insights to improve model performance.
The insights for each task type should contain 4 unique insights.
Be specific about models' choices, ensemble and tuning techniques, and preprocessing & feature engineering techniques.
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


# add role
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