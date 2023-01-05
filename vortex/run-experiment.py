import json
from core import *

f = open("exp.json", "r")
experiment_config = json.load(f)
f.close()

ExperimentLauncher.run(experiment_config)