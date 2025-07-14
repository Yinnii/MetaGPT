import json, requests

from metagpt.ext.sela.utils import clean_json_from_rsp, load_data_config, mcts_logger
from metagpt.llm import LLM

DATA_CONFIG = load_data_config()


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

# Information
{hyperparameter_settings_examples}

You are a Machine Learning Expert tasked with analyzing a dataset. 
You have the following settings from similar datasets that have been used in previous runs.
Use only this information and dataset description to generate insights for different task types in the machine learning pipeline.
Use the software in the information to generation model training insights.
Use the hyperparametersettings defined in the examples to propose Hyperparameter optimization insights for the following task types: 

# Format
```json
[
    {{
        "task_type": "EDA",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
        ]   
    }},
    {{
        "task_type": "Data Preprocessing",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
        ]   
    }},
    {{
        "task_type": "Feature Engineering",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
        ]   
    }},
    {{
        "task_type": "Model Training",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
        ]   
    }},
    {{
        "task_type": "Hyperparameter Optimization",
        "insights": [
            "insight1",
            "insight2",
            "insight3",
        ]
    }},
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
    }},
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


KEY_DATASET_FEATURES = [
    "NumberOfClasses",
    "NumberOfFeatures",
    "NumberOfInstances",
    "NumberOfInstancesWithMissingValues",
    "NumberOfMissingValues",
    "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures",
]

TASK_TO_ID = {"EDA": 1, "Data Preprocessing": 2, "Feature Engineering": 3, "Model Training": 4, "Hyperparameter Tuning":5, "Model Evaluation": 6}


class OntoRAGSolutionDesigner:
    data_dir: str = DATA_CONFIG["datasets_dir"]

    async def generate_solutions(self, dataset_info, dataset_name, save_analysis_pool=True):
        llm = LLM()
        if type(dataset_info) == dict:
            description_prompt = DATASET_DESCRIPTION_SELA_PROMPT.format(
                dataset=dataset_info["description"],
                metadata=self.metadata_builder(dataset_info["metadata"]),
                head=dataset_info["df_head"],
            )
        else:
            description_prompt = DATASET_DESCRIPTION_CUSTOM_PROMPT.format(dataset_description=dataset_info)

        try:
            url = "http://localhost:6666/retrieve_runs"
            payload = {"query": description_prompt}
            examples = requests.post(url, json=payload).json().get("message", "No examples found.")

            context0 = INFORMATION_BASED_INSIGHTS.format(
                description=description_prompt, 
                hyperparameter_settings_examples=examples
            )

            insights0 = await llm.aask(context0)
            
            mcts_logger.info(f"Retrieved first insight example: {insights0}")

        except Exception as e:
            print(f"Error retrieving parameters: {e}")
            examples = "No examples found. Please provide hyperparameter settings manually."
        
        context = DATASET_INSIGHT_PROMPT_ONTORAG.format(description=description_prompt, 
                                                        hyperparameter_settings_examples=insights0)
        rsp = await llm.aask(context)
        rsp = clean_json_from_rsp(rsp)
        analysis_pool = self.process_analysis_pool(json.loads(rsp))
        if save_analysis_pool:
            dataset_path = f"{self.data_dir}/{dataset_name}"
            self.save_analysis_pool(dataset_path, analysis_pool)
        return analysis_pool

    async def propose_new_insights(self, solution, score):
        llm = LLM()
        context = INSIGHT_PROPOSAL_PROMPT_ONTORAG.format(solution_code=solution, dev_score=score)
        rsp = await llm.aask(context)
        rsp = clean_json_from_rsp(rsp)
        new_insights = self.process_analysis_pool(json.loads(rsp))
        return new_insights

    def process_analysis_pool(self, insights_rsp):
        analysis_pool = []
        for task_type_insights in insights_rsp:
            task_type = task_type_insights["task_type"]
            for insight in task_type_insights["insights"]:
                analysis_pool.append({"Analysis": insight, "Category": task_type, "task_id": TASK_TO_ID[task_type]})
        return analysis_pool

    def metadata_builder(self, qualities):
        metadata = {}
        for key in KEY_DATASET_FEATURES:
            metadata[key] = qualities.get(key, "N/A")
        metadata_text = json.dumps(metadata, indent=4)
        return metadata_text

    def save_analysis_pool(self, dataset_path, analysis_pool):
        fpath = f"{dataset_path}/ds_analysis_pool.json"
        with open(fpath, "w") as file:
            json.dump(analysis_pool, file, indent=4)
