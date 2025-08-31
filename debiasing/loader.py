import json

def load_def_set_pairs(json_filepath):
	with open(json_filepath, "r") as f:
		loadedData = json.load(f)
		def_sets = loadedData.get("definite_sets", [])
		pairs = {}

		for i, values in enumerate(def_sets):
			pairs[i] = []
			for v1 in values:
				for v2 in values:
					s = set([v1, v2])
					if len(s) > 1 and not listContainsMultiple([v1, v2], pairs[i]):
						pairs[i].append([v1, v2])

		res = []
		for j in range(len(pairs[0])):
			res.append({k: pairs[k][j] for k in pairs})
		return res

def load_analogy_templates(json_filepath, mode):
	with open(json_filepath, "r") as f:
		loadedData = json.load(f)
		return loadedData.get("analogy_templates", {}).get(mode, {})

def load_eval_terms(json_filepath, mode):
	with open(json_filepath, "r") as f:
		loadedData = json.load(f)
		return loadedData.get("eval_targets", []), loadedData.get("analogy_templates", {}).get(mode, {}).values()

def load_def_sets(json_filepath):
	with open(json_filepath, "r") as f:
		loadedData = json.load(f)
		return {i: v for i, v in enumerate(loadedData.get("definite_sets", []))}