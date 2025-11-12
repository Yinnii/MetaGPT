import subprocess
from metagpt.actions import Action
from metagpt.utils.common import CodeParser
from metagpt.logs import logger
import os
import psycopg2
import py2nb
from datetime import datetime
from metagpt.const import ROLE_PATH

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "openml_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "yin-postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5434")

connection = psycopg2.connect(
    dbname="openml",
    user="postgres",
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT
)
cursor = connection.cursor()

class RunMLCode(Action):
    name: str = "RunMLCode"
    data_dir: str = "/home/yin/Projects/MetaGPT/metagpt/ext/sela/SELA_datasets"
    role_dir: str = ROLE_PATH
    success: bool = False

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
          parsed_code = CodeParser.parse_code(block=None, text=code_text, lang="python")
          path = self._store_code(parsed_code, dataset)
        except Exception as e:
          logger.error(f"Error parsing or storing ML code: {e}")
          path = self._store_code(code_text, dataset)

        code_result = await self.execute_code(path)

        if self.success:
            cursor.execute("""CREATE TABLE IF NOT EXISTS dataset_model_training (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset VARCHAR(255),
                code TEXT
            )""")
            cursor.execute("""INSERT INTO dataset_model_training (dataset, code, created_at) VALUES (%s, %s, %s)""", (dataset, code_text, datetime.now()))
            connection.commit()

            # store the code also as jupyter notebook
            # check if Node-0-0-0-0-0.ipynb already exists
            counter = 0
            node_notebook_path = f'{self.role_dir}/{dataset}/Node-0-0-0-0-{counter}.ipynb'

            # increment counter until a non-existing file is found
            while os.path.exists(node_notebook_path):
                logger.info(f"Notebook {node_notebook_path} already exists. Skipping notebook creation.")
                counter += 1
                node_notebook_path = f'{self.role_dir}/{dataset}/Node-0-0-0-0-{counter}.ipynb'

            executed_notebook = py2nb.convert(path, output_name=node_notebook_path)
            logger.info(f"Executed notebook saved at {node_notebook_path}.")

        return code_result
    
    async def execute_code(self, path: str):
        result = subprocess.run(["python3", path], capture_output=True, text=True)
        if result.returncode != 0:
            return result.stderr
        self.success = True
        logger.info(f"Code executed successfully: {result.stdout}")
        return result.stdout   