import json
from metagpt.actions import Action
from metagpt.schema import Message
from metagpt.logs import logger 
from metagpt.prompts.rcr.run_retrieval import RUN_RETRIEVAL_EXAMPLE, RUN_RETRIEVAL_PROMPT

class RetrieveRunInformation(Action):
    """
    An action to retrieve run information from a Jupyter notebook file.
    """
    name: str = "RetrieveRunInformation" 

    async def run(self, notebook_data) -> dict:
        user_prompt = RUN_RETRIEVAL_PROMPT + RUN_RETRIEVAL_EXAMPLE + "Based on the following notebook data. Retrieve the relevant information."

        context = self.llm.format_msg([Message(content=user_prompt + json.dumps(notebook_data), role="user")])
        # logger.info(f"Formatted context for retrieval: {context}")

        rsp = await self.llm.aask(context)

        return rsp