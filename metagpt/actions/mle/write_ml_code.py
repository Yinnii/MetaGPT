from __future__ import annotations

from metagpt.actions import Action
from metagpt.prompts.mle.write_ml_code import (
  ML_SYSTEM_MESSAGE,
  ML_PROMPT
)
from metagpt.prompts.di.write_analysis_code import (
  REFLECTION_PROMPT,
  REFLECTION_SYSTEM_MSG,
  DEBUG_REFLECTION_EXAMPLE,
)
from metagpt.logs import logger

from metagpt.schema import Message, Plan
from metagpt.utils.common import CodeParser, remove_comments


class WriteMLCode(Action):
    
    name: str = "WriteMLCode"

    async def _debug_with_reflection(self, context: list[Message], working_memory: list[Message], target_column: str):
        reflection_prompt = REFLECTION_PROMPT.format(
            debug_example=DEBUG_REFLECTION_EXAMPLE,
            context=context,
            previous_impl=working_memory,
            target_column = target_column
            )

        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])
        # reflection = json.loads(CodeParser.parse_code(block=None, text=rsp))
        # return reflection["improved_impl"]
        reflection = CodeParser.parse_code(block=None, text=rsp)
        return reflection

    async def run(
        self,
        configuration: str,
        working_memory: list[Message] = None,
        use_reflection: bool = False,
        target_column: str = "class",
        dataset_name: str = None,
        **kwargs,
    ) -> str:

        structual_prompt = ML_PROMPT.format(
            dataset_name=dataset_name,
            target_column=target_column,
            configuration=configuration
        )

        working_memory = working_memory or []
        context = self.llm.format_msg([Message(content=structual_prompt, role="user")] + working_memory)

        # LLM call
        if use_reflection:
            code = await self._debug_with_reflection(context=context, working_memory=working_memory, target_column=target_column)
        else:
            rsp = await self.llm.aask(context, system_msgs=[ML_SYSTEM_MESSAGE], **kwargs)
            code = CodeParser.parse_code(block=None, text=rsp)

        return code
