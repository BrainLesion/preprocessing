import os
from .core import brats_preprocessing

DATA = os.getenv("DATA")
LOGS = os.getenv("LOGS")
MODULES = os.getenv("MODULE")

def worker(jobName, headers, params, added_params, **kwargs):
	
	jodData = added_params.get(jobName, {})
	input_dict = jodData.get("input", {})
	output_dict = jodData.get("output", {})

	for k in input_dict:
		input_dict[k] = os.path.join(DATA, input_dict[k])

	for k in output_dict:
		output_dict[k] = os.path.join(DATA, output_dict[k])

	if "baseDir" in output_dict:
		if not os.path.isdir(output_dict.get("baseDir")):
			try:
				os.makedirs(output_dict.get("baseDir"), mode=0o777)
			except:
				pass

	brats_preprocessing(input_dict=input_dict, output_dict=output_dict, options_dict=params)
