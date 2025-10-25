import asyncio
import json, json_repair
from metagpt.context import Context
from metagpt.roles.role import Role
from metagpt.logs import logger
from metagpt.actions.rcr.retrieveInformation import RetrieveRunInformation
from metagpt.schema import Message
from .utils import clean_json_from_rsp

RUN_RETRIEVAL_PROMPT = """
You are a helpful assistant that retrieves machine learning run information from Jupyter notebooks. 
Provide relevant information from the notebook.
"""

class RunContextRetriever(Role):
    name: str = "Run Context Retriever"
    profile: str = "RunContextRetriever"
    goal: str = "Retrieve machine learning run information from Jupyter notebooks."
    description: str = "A role that retrieves machine learning run information from Jupyter notebooks."
    constraints: str = "You must return the information in a valid JSON format."
    max_react_loop: int = 2
    retrieval_prompt: str = RUN_RETRIEVAL_PROMPT

    def __init__(self, notebook_path, dataset, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([RetrieveRunInformation]) 
        self._set_react_mode(react_mode="react", max_react_loop=1)
        self.notebook_path = notebook_path
        self.dataset = dataset
        self.count = 0
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            self.notebook_data = json.load(f)

        # clean the notebook data and reduce the context by doing so. 
        for cell in self.notebook_data.get("cells", []):
            if "outputs" in cell:
                cell["outputs"] = [
                    o for o in cell["outputs"]
                    if not (o.get("output_type") == "stream" and o.get("name") == "stderr")
                ]

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}")
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]

        if todo.name == "RetrieveRunInformation":
            try:
                # Run the action and log the raw result
                result = await todo.run(self.notebook_data)
                logger.info(f"Raw result from todo.run: {result}")

                # Clean the result and log the cleaned output
                result_cleaned = clean_json_from_rsp(result)
                logger.info(f"Cleaned JSON: {result_cleaned}")

                # Fallback to raw result if cleaning fails
                if result_cleaned is None:
                    logger.error("Failed to clean JSON. Using raw result.")
                    result_cleaned = result

                # Attempt to parse the cleaned result as JSON
                try:
                    if type(result_cleaned) is str:
                      result_dict = json.loads(result_cleaned)
                      result_dict["dataset"] = {"name": self.dataset}
                      result = json.dumps(result_dict)
                    elif type(result_cleaned) is dict:
                      result_cleaned["dataset"] = {"name": self.dataset}
                      result = json.dumps(result_cleaned)
                    else:
                      result = json.dumps({
                          "error": "Unexpected format",
                          "details": "The cleaned result is neither a string nor a dictionary."
                      })
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cleaned result as JSON: {e}")

                # Create the message with the final result
                msg = Message(content=result, role="assistant", cause_by=type(todo))
                return msg

            except Exception as e:
                # Catch any unexpected errors and log them
                logger.error(f"Unexpected error in RetrieveRunInformation: {e}")
                result = json.dumps({
                    "error": "Unexpected error",
                    "details": str(e)
                })
                msg = Message(content=result, role="assistant", cause_by=type(todo))
                return msg
        else:
            return msg

# async def main():
#     context = Context()
#     role = RunContextRetriever(context=context, notebook_path="/home/yin/Data/SELA_4o_mini/creditg/mcts-creditg_202507191737_dev_best.ipynb", dataset="creditg")
#     result = await role.run(RUN_RETRIEVAL_PROMPT)

# asyncio.run(main())