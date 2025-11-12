import json, requests, re

from json_repair import repair_json
from metagpt.ext.sela.machine_learning_expert import MachineLearningExpert
from metagpt.context import Context

from metagpt.ext.sela.utils import clean_json_from_rsp, load_data_config, mcts_logger
from metagpt.llm import LLM

from metagpt.prompts.ontorag import (
    DATASET_DESCRIPTION_SELA_PROMPT,
    DATASET_DESCRIPTION_CUSTOM_PROMPT,
    INFORMATION_BASED_INSIGHTS,
    INSIGHT_PROPOSAL_PROMPT_ONTORAG,
    MALEX_RUN_PROMPT,
    INSIGHT_PROPOSER,
    HYPERPARAMETER_PROPOSER,
    INSIGHT_PROPOSER_EXAMPLE,
    HYPERPARAMETER_PROPOSER_EXAMPLE,
)

DATA_CONFIG = load_data_config()

KEY_DATASET_FEATURES = [
    "NumberOfClasses",
    "NumberOfFeatures",
    "NumberOfInstances",
    "NumberOfInstancesWithMissingValues",
    "NumberOfMissingValues",
    "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures",
]

TASK_TO_ID = {"EDA": 1, "Data Preprocessing": 2, "Feature Engineering": 3, "Model Training": 4, "Hyperparameter Optimization": 5, "Model Evaluation": 6}

class OntoRAGSolutionDesigner:
    data_dir: str = DATA_CONFIG["datasets_dir"]

    def __init__(self, with_pretraining: bool):
        self.with_pretraining = with_pretraining

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

        url = "http://localhost:6666/retrieve_runs"
        payload = {"query": f"Retrieve runs for the dataset {dataset_name}" + description_prompt}
        examples = requests.post(url, json=payload).json().get("message", "No examples found.")
        # examples = clean_json_from_rsp(examples)

        mcts_logger.info(f"Retrieved examples: {examples} with type {type(examples)}")

        if isinstance(examples, str):
            examples = json.loads(examples)
        
        if isinstance(examples, dict):
            try:
                runs = examples.get("runs", [])
            except Exception as e:
                mcts_logger.error(f"Error retrieving runs from examples: {e}")
                runs = examples
        elif isinstance(examples, list):
            runs = examples

        dataset_name = dataset_info.get("dataset_name", dataset_name)
        target_name = dataset_info.get("target_col", "class")
        mcts_logger.info(f"Dataset is {dataset_name} with target name: {target_name}")

        results = []
        if self.with_pretraining:
          mcts_logger.info(f"Extract scores using malex.")
          results = await self.retrieve_run_results(runs, dataset_name, target_name)
          
        predefined_insights = await self.create_pre_insights(examples, results)

        mcts_logger.info(f"Predefined insights: {predefined_insights} with type {type(predefined_insights)}")

        context0 = INFORMATION_BASED_INSIGHTS.format(
            description=description_prompt, 
            predefined_insights=predefined_insights
        )

        insights = await llm.aask(context0)
        insights = clean_json_from_rsp(insights)

        insights_json = json.loads(insights)
        insights = self.append_new_insights_to_predefined(predefined_insights, insights_json)

        analysis_pool = self.process_analysis_pool(insights)

        if save_analysis_pool:
            dataset_path = f"{self.data_dir}/{dataset_name}"
            self.save_analysis_pool(dataset_path, analysis_pool)
            mcts_logger.info(f"Analysis pool saved to {dataset_path}/ds_analysis_pool.json")
    
        return analysis_pool

    # Retrieve the runs from the ontologyRAG and iterate over the runs to retrieve the results for each run with the configurations
    async def retrieve_run_results(self, runs, dataset_name: str, target_name: str = "class"):
        results = []
        for run in runs: 
            context = Context()
            # if dataset_name is None:
            #     dataset_name = run.get("dataset").get("dataset_name")
            mcts_logger.info(f"Running configuration: {run} for dataset: {dataset_name}")
            malex = MachineLearningExpert(context=context, dataset=dataset_name, target_column=target_name)
            query = MALEX_RUN_PROMPT.format(configuration=run)
            mcts_logger.info(f"Running query: {query}")
            result = await malex.run(query)
            mcts_logger.info(f"Run result: {result} with type {type(result)}")
            if result is None:
                mcts_logger.error(f"Run result is None for configuration: {run}")
                results.append({"train_score": 0, "test_score": 0, "score": 0})
            else:
                result = result.content
                # clear results
                mcts_logger.info(f"Processing run result: {result}")
                try:
                  if isinstance(result, str):
                      result = json.loads(result)
                  result.pop("model_path", None)  # remove model path if exists
                  results.append(result)
                except Exception as e:
                  results.append({"train_score": result.get("train_score", 0), "test_score": result.get("test_score", 0), "score": result.get("test_score", 0)})
        return results

    async def create_pre_insights(self, examples, results) -> list:
        llm = LLM()
        predefined_insights = []

        if not results or len(results) == 0:
            predefined_insights.append({
                "task_type": "EDA",
                "insights": ["Perform exploratory data analysis on the training, dev, and test datasets."],
            })
            predefined_insights.append({
                "task_type": "Data Preprocessing",
                "insights": ["Preprocess the train, dev, and test datasets, including handling missing values and encoding categorical variables."],
            })
            predefined_insights.append({
                "task_type": "Feature Engineering",
                "insights": ["Engineer features for the training, dev, and test datasets to improve model performance."],
            })
            predefined_insights.append({
                "task_type": "Model Training",
                "insights": [],
            })
            predefined_insights.append({
                "task_type": "Hyperparameter Optimization",
                "insights": [],
            })

            if isinstance(examples, dict):
                examples = examples.get("runs", [])
            elif not isinstance(examples, list):
                examples = [examples]

            for example in examples:
                # if isinstance(example, str):
                #     example = json.loads(example)
                mcts_logger.info(f"Processing example: {type(example)} - {example}")
                try:
                  flow = example.get("run").get("flow", {})
                  mcts_logger.info(f"Flow details: {flow}")
                  clf = re.sub(r'\d+', '', str(flow.get('implementation', '')))
                  software_list = flow.get('software', [])
                  software_str = ', '.join([f"{s['name']} {s['version']}" for s in software_list]) if isinstance(software_list, list) else str(software_list)
                  software = re.sub(r'\d+', '', software_str)

                  train_insight = await llm.aask(INSIGHT_PROPOSER.format(software=software, classifier=clf) + INSIGHT_PROPOSER_EXAMPLE)
                  hyper_insight = await llm.aask(HYPERPARAMETER_PROPOSER.format(hyperparameter_settings=flow.get('hyperparametersettings', {})) + HYPERPARAMETER_PROPOSER_EXAMPLE)
                  
                  mcts_logger.info(f"Generated insights: {train_insight}, {hyper_insight}")

                  predefined_insights[3]["insights"].append(train_insight)
                  predefined_insights[4]["insights"].append(hyper_insight)
                except Exception as e:
                  mcts_logger.error(f"Error processing example: {e}")
                  other_runs = example.get("runs")
                  for r in other_runs:
                      mcts_logger.info(f"Processing example: {type(example)} - {example}")
                      try:
                        flow = r.get("run").get("flow", {})
                        mcts_logger.info(f"Flow details: {flow}")
                        clf = re.sub(r'\d+', '', str(flow.get('implementation', '')))
                        software_list = flow.get('software', [])
                        software_str = ', '.join([f"{s['name']} {s['version']}" for s in software_list]) if isinstance(software_list, list) else str(software_list)
                        software = re.sub(r'\d+', '', software_str)

                        train_insight = await llm.aask(INSIGHT_PROPOSER.format(software=software, classifier=clf) + INSIGHT_PROPOSER_EXAMPLE)
                        hyper_insight = await llm.aask(HYPERPARAMETER_PROPOSER.format(hyperparameter_settings=flow.get('hyperparametersettings', {})) + HYPERPARAMETER_PROPOSER_EXAMPLE)
                        
                        mcts_logger.info(f"Generated insights: {train_insight}, {hyper_insight}")

                        predefined_insights[3]["insights"].append(train_insight)
                        predefined_insights[4]["insights"].append(hyper_insight)
                      except Exception as e:
                          mcts_logger.error(f"Something went wrong: {e}")
                  continue
            mcts_logger.info(f"Predefined insights after processing: {predefined_insights}")
          
            return predefined_insights
        else:             
            predefined_insights.append({
                "task_type": "EDA",
                "insights": ["Perform exploratory data analysis on the training, dev, and test datasets."],
                "score": [{"train_score": 0, "test_score": 0, "score": 0}]
            })
            predefined_insights.append({
                "task_type": "Data Preprocessing",
                "insights": ["Preprocess the train, dev, and test datasets, including handling missing values and encoding categorical variables."],
                "score": [{"train_score": 0, "test_score": 0, "score": 0}]
            })
            predefined_insights.append({
                "task_type": "Feature Engineering",
                "insights": ["Engineer features for the training, dev, and test datasets to improve model performance."],
                "score": [{"train_score": 0, "test_score": 0, "score": 0}]
            })
            predefined_insights.append({
                "task_type": "Model Training",
                "insights": [],
                "score": []
            })
            predefined_insights.append({
                "task_type": "Hyperparameter Optimization",
                "insights": [],
                "score": []
            })

            if isinstance(examples, dict):
                examples = examples.get("runs", [])
            elif not isinstance(examples, list):
                examples = [examples]

            for example in examples:
                # if isinstance(example, str):
                #     example = json.loads(example)

                mcts_logger.info(f"Processing example: {type(example)} - {example}")
                try:
                  flow = example.get("run").get("flow", {})
                  mcts_logger.info(f"Flow details: {flow}")
                  clf = re.sub(r'\d+', '', str(flow.get('implementation', '')))
                  software_list = flow.get('software', [])
                  software_str = ', '.join([f"{s['name']} {s['version']}" for s in software_list]) if isinstance(software_list, list) else str(software_list)
                  software = re.sub(r'\d+', '', software_str)
                  train_insight = await llm.aask(INSIGHT_PROPOSER.format(software=software, classifier=clf) + INSIGHT_PROPOSER_EXAMPLE)
                  hyper_insight = await llm.aask(HYPERPARAMETER_PROPOSER.format(hyperparameter_settings=flow.get('hyperparametersettings', {})) + HYPERPARAMETER_PROPOSER_EXAMPLE)
                  
                  mcts_logger.info(f"Generated insights: {train_insight}, {hyper_insight}")

                  predefined_insights[3]["insights"].append(train_insight)
                  predefined_insights[4]["insights"].append(hyper_insight)
                except Exception as e:
                  mcts_logger.error(f"Error processing example: {e}")
                  continue

            for result in results:
                if isinstance(result, str):
                    result = json.loads(result)
                mcts_logger.info(f"Processing result: {result}")
                if result['test_score'] == 1.0:
                    result['test_score'] = 0 # A score of 1.0 indicates leakage or overfitting, set to 0 to avoid misleading the MCTS
                predefined_insights[3]["score"].append({"train_score": result["train_score"], "test_score": result["test_score"], "score": result['test_score']})
                predefined_insights[4]["score"].append({"train_score": result["train_score"], "test_score": result["test_score"], "score": result['test_score']})

                if result["test_score"] > predefined_insights[2]["score"][0]["test_score"]:
                    predefined_insights[2]["score"][0]["test_score"] = result["test_score"]
                    predefined_insights[2]["score"][0]["train_score"] = result["train_score"]
                    predefined_insights[2]["score"][0]["score"] = result['test_score']
                    predefined_insights[1]["score"][0]["test_score"] = result["test_score"]
                    predefined_insights[1]["score"][0]["train_score"] = result["train_score"]
                    predefined_insights[1]["score"][0]["score"] = result['test_score']
                    predefined_insights[0]["score"][0]["test_score"] = result["test_score"]
                    predefined_insights[0]["score"][0]["train_score"] = result["train_score"]
                    predefined_insights[0]["score"][0]["score"] = result['test_score']

            mcts_logger.info(f"Predefined insights after processing: {predefined_insights}")
            return predefined_insights

    def append_new_insights_to_predefined(self, predefined_insights, new_insights) -> list:
        for task_type_insights in new_insights:
            task_type = task_type_insights["task_type"]
            for insight in task_type_insights["insights"]:
                found = False
                for predefined in predefined_insights:
                    if predefined["task_type"] == task_type:
                        predefined["insights"].append(insight)
                        found = True
                        break
                if not found:
                    predefined_insights.append({
                        "task_type": task_type,
                        "insights": [insight]
                    })
        mcts_logger.info(f"New insights appended.")
        return predefined_insights

    async def propose_new_insights(self, solution, score):
        llm = LLM()
        context = INSIGHT_PROPOSAL_PROMPT_ONTORAG.format(solution_code=solution, dev_score=score)
        rsp = await llm.aask(context)
        rsp = clean_json_from_rsp(rsp)
        new_insights = self.process_analysis_pool(json.loads(rsp))
        return new_insights

    # !!! Extending the previous analysis pool from SELA and adds the existing scores and insights already to the pool for later usage.
    # By doing so, we can initialize an existing tree for warm start/ hot start MCTS.
    def process_analysis_pool(self, insights_rsp):
        mcts_logger.info(f"Processing analysis pool: {insights_rsp} with type {type(insights_rsp)}")
        if not isinstance(insights_rsp, list):
            insights_rsp = json.loads(insights_rsp)

        analysis_pool = []

        for task_type_insights in insights_rsp:
            task_type = task_type_insights["task_type"]
            insights = task_type_insights["insights"]
            scores = task_type_insights.get("score", [])
            for i, insight in enumerate(insights):
                if self.with_pretraining:
                    if i < len(scores):
                        score = scores[i]
                    else:
                        score = {"train_score": 0, "test_score": 0, "score": 0}

                    analysis_pool.append({
                        "Analysis": insight,
                        "Category": task_type,
                        "task_id": TASK_TO_ID[task_type],
                        "Score": score
                    })
                else:
                    analysis_pool.append({
                        "Analysis": insight,
                        "Category": task_type,
                        "task_id": TASK_TO_ID[task_type],
                    })
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
