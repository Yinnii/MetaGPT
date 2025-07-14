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

    async def _debug_with_reflection(self, context: list[Message], working_memory: list[Message]):
        reflection_prompt = REFLECTION_PROMPT.format(
            debug_example=DEBUG_REFLECTION_EXAMPLE,
            context=context,
            previous_impl=working_memory,
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
        **kwargs,
    ) -> str:

        structual_prompt = ML_PROMPT.format(
            configuration=configuration
        )

        working_memory = working_memory or []
        context = self.llm.format_msg([Message(content=structual_prompt, role="user")] + working_memory)

        # LLM call
        if use_reflection:
            code = await self._debug_with_reflection(context=context, working_memory=working_memory)
        else:
            rsp = await self.llm.aask(context, system_msgs=[ML_SYSTEM_MESSAGE], **kwargs)
            code = CodeParser.parse_code(block=None, text=rsp)

        return code
