from __future__ import annotations

import asyncio, re, json

from metagpt.context import Context
from metagpt.actions.mle.write_ml_code import WriteMLCode
from metagpt.actions.mle.run_ml_code import RunMLCode
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message


class MachineLearningExpert(Role):
    name: str = "Malex"
    profile: str = "MachineLearningExpert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([WriteMLCode, RunMLCode])
        self._set_react_mode(react_mode="by_order", max_react_loop=5)

    async def _act(self) -> Message:
      logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
      # By choosing the Action by order under the hood
      # todo will be first SimpleWriteCode() then SimpleRunCode()
      todo = self.rc.todo

      msg = self.get_memories(k=1)[0]  # find the most k recent messages

      if todo.name == "WriteMLCode":
        msg.content = await self._parse_configuration(msg)
        result = await todo.run(msg.content)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
      elif todo.name == "RunMLCode":
        print(f"Running ML code for dataset: {self.dataset}")
        result = await todo.run(msg.content, dataset=self.dataset)
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)

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