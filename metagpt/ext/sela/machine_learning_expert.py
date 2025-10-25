from __future__ import annotations

import asyncio, re, json
from json_repair import repair_json

from metagpt.actions.mle.write_ml_code import WriteMLCode
from metagpt.actions.mle.run_ml_code import RunMLCode
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

from metagpt.prompts.mle.write_ml_code import (
    REFLECT_PROMPT
)

class MachineLearningExpert(Role):
    name: str = "Malex"
    profile: str = "MachineLearningExpert"
    goal: str = """You are a Machine Learning expert. 
                  Your task is to create machine learning models based on user requirements. 
                  You will write code to create the model and run the code to evaluate the model. 
                  You will iterate this process until you have a satisfactory model."""
    constraints: str = """You must use Python and common ML libraries such as scikit-learn, TensorFlow, or PyTorch. 
                          You must write code that is efficient and follows best practices.
                          You must handle errors gracefully and provide meaningful feedback.
                          You must not use any external APIs or services."""
    max_react_loop: int = 3  # used here to reflect

    def __init__(self, dataset, target_column, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteMLCode, RunMLCode] * self.max_react_loop) 
        self._set_react_mode(react_mode="by_order", max_react_loop=5)
        self.dataset = dataset
        self.target_column = target_column
        self.count = 0  # used to count the number of react loops
        self.finished = False

    async def _act(self) -> Message:
      logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
      # By choosing the Action by order under the hood
      # todo will be first SimpleWriteCode() then SimpleRunCode()
      todo = self.rc.todo

      msg = self.get_memories(k=1)[0]  # find the most k recent messages

      if self.finished:
        logger.info("Finished all tasks, exiting.")
        return msg

      if todo.name == "WriteMLCode":
        if msg.role != "assistant":
          logger.info(f"Message is {msg}")
          msg.content = await self._parse_configuration(msg)
        result = await todo.run(msg.content, working_memory=self.rc.memory.get(), dataset_name=self.dataset, target_column=self.target_column)
        msg = Message(content=result, role="assistant", cause_by=type(todo))
        self.rc.memory.add(msg)
        
      elif todo.name == "RunMLCode":
        result = await todo.run(msg.content, dataset=self.dataset)
        msg = Message(content=result, role="assistant", cause_by=type(todo))
        logger.info(f"RunMLCode result: {result}")
        # if message contains error, reflect on it by writing a new code
        if "Error" in result or "error" in result.lower():
            reflect_msg = REFLECT_PROMPT.format(
                feedback = result,
                dataset_name = self.dataset,
                target_column = self.target_column
            )
            msg.content = reflect_msg
            msg.role = "assistant"
            msg.cause_by = "RunMLCode"
            self.count += 1
            if self.count > self.max_react_loop:
                msg.content = "Error running the configuration and create a model."
                self.finished = True
                self.rc.memory.add(msg)
                return msg
               
        else:
            result_dict = await self._parse_result_to_dict(result)
            msg.content = json.dumps(result_dict, indent=4)
            msg.role = "assistant"
            msg.cause_by = "RunMLCode"
            self.rc.memory.add(msg)
            self.finished = True
            return msg
    
    # parse incoming message and extract the json configuration
    async def _parse_configuration(self, message: Message) -> str:
        message = re.search(r'(\{.*\})', message.content, re.DOTALL)
        
        if message:
            json_str = message.group(1)
            json_str = repair_json(json_str)  # repair the json string if needed
            logger.info(f"Extracted JSON configuration: {json_str}")
            try:
                configuration = json.loads(json_str)

                # # get dataset name
                # try: 
                #   dataset = configuration.get("run", {}).get("dataset", {}).get("dataset_name", "")
                # except Exception as e:
                #   dataset = configuration.get("dataset", {}).get("dataset_name", "")

                # self.dataset = dataset

                return json.dumps(configuration, indent=4)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON configuration: {e}")
                raise ValueError("Invalid JSON format in the message content.")
        else:
            logger.error("No JSON configuration found in the message content.")
            raise ValueError("No JSON configuration found in the message content.")
        
    async def _parse_result_to_dict(self, result: str) -> dict:
        try:
            # search for float values 
            float_values = re.findall(r"[-+]?\d*\.\d+|\d+", result)
            pattern = r"Model saved to (.+)"
            model_path = re.search(pattern, result, re.DOTALL)
            if float_values and len(float_values) >= 2:
                training_accuracy = float(float_values[0])
                model_accuracy = float(float_values[1])
                model_path = model_path.group(1).strip()

                return {
                    "train_score": training_accuracy,
                    "test_score": model_accuracy,
                    "model_path": model_path
                }
            elif float_values and len(float_values) == 1:
                test_accuracy = float(float_values[0])
                model_path = model_path.group(1).strip()

                return {
                    "train_score": -1,
                    "test_score": test_accuracy,
                    "model_path": model_path
                }
                
        except Exception as e:
            logger.error(f"Failed to parse result: {e}")
            # TODO: maybe the score is not in the correct order, what to do then?
            try:
                score = re.search(r"(\d+\.\d+)", result)
                if score:
                    return {"train_score": float(score.group(1)), "test_score": float(score.group(1))}
            except Exception as e:
                logger.error(f"Failed to retrieve score from result: {e}")
                return {"train_score": 0.0, "test_score": 0.0}

# async def main():
#     msg = '''Write and run a python script to train the dataset with the following configuration:
#             {
#             "run": {
#                 "name": "run48443",
#                 "dataset": {
#                     "dataset_name": "mfeatfactors",
#                     "qualities": {
#                         "description": "One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps."
#                     }
#                 },
#                 "flow": {
#                     "implementation": "wekaBaggingLMT48443",
#                     "software": "Weka48443",
#                     "hyperparametersettings": {
#                         "num_slots": "1",
#                         "W": "weka.classifiers.trees.LMT",
#                         "S": "1",
#                         "P": "100",
#                         "I": "10"
#                     }
#                 },
#                 "evaluation": {
#                     "measure": "predictive_accuracy",
#                     "value": 0.984848
#                 }
#               }
#             }
#           '''
#     context = Context()
#     role = MachineLearningExpert(context=context, dataset="mfeatfactors")
#     result = await role.run(msg)
#     logger.info(result)

# asyncio.run(main())