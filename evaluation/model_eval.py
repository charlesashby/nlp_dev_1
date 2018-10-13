import os
import sys

from evaluation.eval import evaluate, test
from models.ngram.ngram_tree_simple import NG_Tree_Simple
from models.ngram.ngram_tree_backoff import NG_Tree_Backoff


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

def evaluate_NGTree(model, pre_wordscount, post_wordscount, lang, max_pred=1, trace=False):
	print(f"{model.__name__}:")
	tree = model(pre_wordscount, post_wordscount)
	tree.train(get_training_file(lang))
	evaluate(tree, pre_wordscount, post_wordscount, lang, words[lang], trace, max_pred)
	print()

if __name__ == '__main__':
	lang = "en"
	load_corrections(lang)
	evaluate_NGTree(NG_Tree_Simple, 3, 1, lang, 3)
	evaluate_NGTree(NG_Tree_Backoff, 3, 1, lang, 3)
