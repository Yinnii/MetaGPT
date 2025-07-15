import subprocess
from metagpt.actions import Action
from metagpt.utils.common import CodeParser
from metagpt.logs import logger

class RunMLCode(Action):
    name: str = "RunMLCode"
    data_dir: str = "/home/yin/Projects/MetaGPT/metagpt/ext/sela/SELA_datasets"

    def _store_code(self, code_text: str, dataset: str):
        path = f"{self.data_dir}/{dataset}/model_training.py"
        with open(path, "w") as f:
            f.write(code_text)
        logger.info(f"ML code stored in {path}")
        return path

    async def run(
          self, 
          code_text: str, 
          dataset: str
        ) -> str:
        try:
          parsed_code = CodeParser.parse_code(block=None, text=code_text)
          path = self._store_code(parsed_code, dataset)
        except Exception as e:
          logger.error(f"Error parsing or storing ML code: {e}")
          path = self._store_code(code_text, dataset)

        code_result = await self.execute_code(path)
        return code_result
    
    async def execute_code(self, path: str):
        result = subprocess.run(["python3", path], capture_output=True, text=True)
        if result.returncode != 0:
            return result.stderr
        return result.stdout



  
    