import os
import sys

from evaluation.eval import evaluate, evaluatecrash, test
from evaluation.generate import generate
from models.ngram.ngram_tree_simple import NG_Tree_Simple
from models.ngram.ngram_tree_backoff import NG_Tree_Backoff
from models.ngram.ngram_regular import NG_Regular


# Holds the correction of the test
words = {"en": {}, "fi": {}}
train = "../train/"

def load_correction(file):
	(test_type, lang) = file.split("-")[-1].split(".")
	print(test_type, end=" ")
	words[lang][test_type] = []
	strtofind = '<unk w="'
	nextstart = len(strtofind) + 1
	with open(file, "r", encoding="utf-16") as correction:
		fdata = correction.read()
		start = 0
		index = fdata.find(strtofind, start)
		while index != -1:
			start = index + nextstart
			word_end = fdata.find('"', start)
			words[lang][test_type].append(sys.intern(fdata[start-1:word_end].lower()))
			index = fdata.find(strtofind, start)

def load_corrections(lang):
	files = os.listdir(test)
	print(f"loading {lang} correction: ", end="")
	for file in files:
		if not file.startswith("t-") and file.endswith(lang):
			load_correction(test+file)
	print()

def get_little_training_file(language):
	return f"{train}little.{language}"

def get_training_file(language):
	return f"{train}train-europarl-v7.fi-en.{language}"

def evaluate_NGM(modelclass, pre_wordscount, post_wordscount, lang, max_pred=1, crash=False,trace=False):
	print(f"{modelclass.__name__} ({pre_wordscount}c{post_wordscount}):")
	model = modelclass(pre_wordscount, post_wordscount)
	model.train(get_training_file(lang))
	if crash:
		evaluatecrash(model, pre_wordscount, post_wordscount, lang, words[lang])
	else:
		evaluate(model, pre_wordscount, post_wordscount, lang, words[lang], trace, crash, max_pred)
	print()

def generate_NGM(modelclass, pre_wordscount, lang):
	model = modelclass(pre_wordscount, 0)
	model.train(get_training_file(lang))
	starts = ["The european", "I would", "This year", "However", "The euro is"]
	for start in starts:
		pre_words = start.strip()
		print(f'Generating with {modelclass.__name__} ("{pre_words}"):')
		sentence = generate(model, pre_words.lower().split())
		print(sentence)

if __name__ == '__main__':
	lang = "en"
	mclass = NG_Tree_Backoff
	load_corrections(lang)
	contexts = [(2, 1)]
	for (pre, post) in contexts:
		print(f"{mclass.__name__} ({pre}c{post}):")
		evaluate_NGM(mclass, pre, post, lang, trace=True, crash=False, max_pred=1)
