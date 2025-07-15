from __future__ import annotations

import asyncio, re, json

from metagpt.context import Context
from metagpt.actions.di.write_analysis_code import WriteAnalysisCode
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
    max_react_loop: int = 3  # used here to reflect

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteMLCode, RunMLCode] * self.max_react_loop) 
        self._set_react_mode(react_mode="by_order", max_react_loop=5)
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
          msg.content = await self._parse_configuration(msg)
        result = await todo.run(msg.content, working_memory=self.rc.memory.get())
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
            msg.content = f"Model trained successfully: {result}."
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
            try:
                configuration = json.loads(json_str)

                # get dataset name
                try: 
                  dataset = configuration.get("run", {}).get("dataset", {}).get("dataset_name", "")
                except Exception as e:
                  dataset = configuration.get("dataset", {}).get("dataset_name", "")

                self.dataset = dataset

                return json.dumps(configuration, indent=4)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON configuration: {e}")
                raise ValueError("Invalid JSON format in the message content.")
        else:
            logger.error("No JSON configuration found in the message content.")
            raise ValueError("No JSON configuration found in the message content.")

async def main():
    msg = '''Write and run a python script to train the dataset with the following configuration:
            {
            "run": {
                "name": "run48443",
                "dataset": {
                    "dataset_name": "mfeatfactors",
                    "qualities": {
                        "description": "One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps."
                    }
                },
                "flow": {
                    "implementation": "wekaBaggingLMT48443",
                    "software": "Weka48443",
                    "hyperparametersettings": {
                        "num_slots": "1",
                        "W": "weka.classifiers.trees.LMT",
                        "S": "1",
                        "P": "100",
                        "I": "10"
                    }
                },
                "evaluation": {
                    "measure": "predictive_accuracy",
                    "value": 0.984848
                }
              }
            }
          '''
    context = Context()
    role = MachineLearningExpert(context=context)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())