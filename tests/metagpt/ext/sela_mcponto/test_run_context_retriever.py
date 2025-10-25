import metagpt.ext.sela.run_context_retriever as rcr
from metagpt.context import Context
from metagpt.logs import logger

if __name__ == "__main__":
    context = Context() 
    role = rcr.RunContextRetriever(context=context, notebook_path="/home/yin/Data/SELA_4o_mini/creditg/mcts-creditg_202507191737_dev_best.ipynb", dataset="creditg")
    result = role.run(rcr.RUN_RETRIEVAL_PROMPT)
    logger.info(f"Test result of RunContextRetriever: {result}")