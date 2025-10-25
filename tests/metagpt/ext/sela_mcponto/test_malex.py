import metagpt.ext.sela.machine_learning_expert as mle
from metagpt.context import Context
from metagpt.logs import logger

if __name__ == "__main__":
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
    role = mle.MachineLearningExpert(context=context, dataset="mfeatfactors")
    result = role.run(msg)
    logger.info(f"Test result of Malex: {result}")