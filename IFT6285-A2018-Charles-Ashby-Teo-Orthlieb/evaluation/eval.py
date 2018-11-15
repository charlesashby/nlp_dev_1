import datetime
import os
import time
from sys import getsizeof

import yaml

from models.mask_reader import NMask

test = "../data/test/"
results = "../evals/stats/"
total_topredict = 0
total_prediction = 0
total_correct = None
total_sentence = 0
total_sentence_correct = 0
is_sentence_correct = None
ans = []
sentence = ""
nbans = 0

def fread(testfile):
	try:
		f = open(testfile, "r", encoding="utf-8")
		return f.read(),"utf-8"
	except UnicodeDecodeError:
		try:
			f = open(testfile, "r", encoding="utf-16")
			return f.read(), "utf-16"
		except UnicodeError:
			f = open(testfile, "r", encoding="iso-8859-1")
			return f.read(), "iso-8859-1"

def get_ans():
	global ans, nbans
	nbans += 1
	txt = ''
	predictions = iter(ans[nbans-1])
	txt += f'{next(predictions)[0]}'
	for pred in predictions:
		txt += f'|{pred[0]}'
	return f'<unk w="{txt}"/>'

def is_pred_correct(predictions, word):
	for i, prediction in enumerate(predictions):
		if prediction[0] == word:
			return i
	return -1

def write_ans(testfile, test_type, lang, encoding):
	with open(testfile, "r", encoding=encoding) as test:
		data = test.read()
	with open(f"{results}ans-{test_type}.{lang}", "w", encoding=encoding) as answer:
		answer.write(' '.join([get_ans() if word.endswith("<unk/>") else word for word in data.split(" ")]))

def _test(model, reader, i, correction, trace, max_pred):
	global total_topredict, total_prediction, total_correct, total_sentence, total_sentence_correct, is_sentence_correct, ans
	if reader.words[i] == "<unk/>":
		# Make a prediction with the context
		pred = model.predict([reader.words[j] for j in range(i)], [reader.words[j] for j in range(i + 1, reader.n)], max_pred)
		if len(pred) == 0:
			# The prediction is empty
			if trace:
				ans.append("")
		else:
			# pred = [("wordA", confidence), ("wordB, confidence), ...]
			# We replace <unk/> with the best prediction in the text
			reader.words[i] = pred[0][0]
			# used to produce a file with all the predictions
			if trace:
				ans.append(pred)
			total_prediction += 1
			# Check if one of the prediction was correct
			pred_correct = is_pred_correct(pred, correction[total_topredict])
			if pred_correct != -1:
				# If it's the first prediction of the sentence
				if is_sentence_correct is None:
					# The sentence is correct (for now)
					is_sentence_correct = True
				total_correct[pred_correct] += 1
			else:
				# The sentence is incorrect
				is_sentence_correct = False
		# One more prediction to be made
		total_topredict += 1
	elif reader.words[i] == ".":
		# Check if the sentence was correct
		if is_sentence_correct:
			total_sentence_correct += 1
		# One more sentence done
		total_sentence += 1
		# Reset this for the next sentence
		is_sentence_correct = None

def _testcrash(model, reader, i, correction):
	global total_topredict, sentence, is_sentence_correct, total_correct
	if reader.words[i] == "<unk/>":
		# Make a prediction with the context
		pred = model.predict([reader.words[j] for j in range(i)], [reader.words[j] for j in range(i + 1, reader.n)], 100)
		if len(pred) > 0:
			# pred = [("wordA", confidence), ("wordB, confidence), ...]
			# We replace <unk/> with the best prediction in the text
			reader.words[i] = pred[0][0]
			# Check if one of the prediction was correct
			pred_correct = is_pred_correct(pred, correction[total_topredict])
			if pred_correct != -1:
				is_sentence_correct = True
				sentence += f"_{pred[0][0]}_ "
			else:
				is_sentence_correct = False
				predlist10 = [elem[0] for elem in pred][:min(len(pred), 10)]
				top10 = "|".join(predlist10)
				sentence += f"_{top10}_({correction[total_topredict]}) "
		# One more prediction to be made
		total_topredict += 1
	elif reader.words[i] == ".":
		if is_sentence_correct == False:
			sentence += reader.words[i]
			print(sentence)
		sentence = ""
		is_sentence_correct = None
	else:
		sentence += reader.words[i] + " "

def _evaluate(model, reader, correction, trace, max_pred):
	global total_topredict, total_prediction, total_correct, total_sentence, total_sentence_correct, is_sentence_correct, ans, nbans
	# Reset the stats and results
	result = {"total_acc": None, "sentence_acc": None, "ans_rate": None}
	total_topredict = 0
	total_prediction = 0
	total_correct = max_pred*[0]
	total_sentence = 0
	total_sentence_correct = 0
	is_sentence_correct = None
	if trace:
		ans = []
		nbans = 0

	starttime = time.time()
	# Check if there's an unknown word before the pre word
	for i in range(reader.pre_words):
		_test(model, reader, i, correction, trace, max_pred)

	# Iterate through the whole text
	while reader.e < len(reader.data):
		_test(model, reader, reader.pre_words, correction, trace, max_pred)
		reader.next_token()

	# Check if there's an unknown word after
	for i in range(reader.pre_words, reader.n):
		_test(model, reader, i, correction, trace, max_pred)

	total_time = time.time() - starttime
	# Compute the results
	result["total_acc"] = sum(total_correct)/total_topredict
	for i, correct in enumerate(total_correct):
		result[f"{i+1}_acc"] = correct/total_topredict
	result["sentence_acc"] = total_sentence_correct/total_sentence
	result["ans_rate"] = total_prediction/total_topredict
	result["time"] = total_time
	return result

def _evaluatecrash(model, reader, correction):
	global total_topredict
	total_topredict = 0

	# Check if there's an unknown word before the pre word
	for i in range(reader.pre_words):
		_testcrash(model, reader, i, correction)

	# Iterate through the whole text
	while reader.e < len(reader.data):
		_testcrash(model, reader, reader.pre_words, correction)
		reader.next_token()

	# Check if there's an unknown word after
	for i in range(reader.pre_words, reader.n):
		_testcrash(model, reader, i, correction)

def evaluate(model, pre_wordscount, post_wordscount, lang, correction, trace, crash, max_pred):
	size = f"# Size of model in memory: {model.sizemo} Mo\n\n"
	files = os.listdir(test)
	stats = {}
	if trace:
		tracetxt = "(trace on)"
	else:
		tracetxt = ""
	# Evaluate the model on every test files
	print(f">>> Evaluating {tracetxt}: ", end="")
	for file in files:
		if file.startswith("t-") and file.endswith(lang):
			test_type = file.split("-")[-1].split(".")[0]
			testfile = f"{test}{file}"
			print(test_type, end=" ")
			(content, encoding) = fread(testfile)
			reader = NMask(content, pre_wordscount, post_wordscount)
			stats[test_type] = _evaluate(model, reader, correction[test_type], trace, max_pred)
			if trace:
				write_ans(testfile, test_type, lang, encoding)

	now = datetime.datetime.now()
	# Write the results in a file (1 file per model)
	modelname = model.__class__.__name__
	with open(f"{results}{modelname}_{lang}_{pre_wordscount}c{post_wordscount}_{max_pred}p.txt", "w") as result:
		desc = f"# Results for {modelname} in {lang} with {max_pred} predictions and this context: {'_ '*pre_wordscount}<unk/> {'_ '*post_wordscount}\n"
		date = f"\n# {now.day}/{now.month}/{now.year}  {now.hour}:{now.minute}:{now.second}"
		result.write(desc+size+yaml.dump(stats, default_flow_style=False)+date)
	print()

def evaluatecrash(model, pre_wordscount, post_wordscount, lang, correction):
	(content, encoding) = fread(f"{test}t-unk-europarl-v7.fi-en-u05.{lang}")
	reader = NMask(content, pre_wordscount, post_wordscount)
	_evaluatecrash(model, reader, correction["u05"])
