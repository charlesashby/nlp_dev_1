from models.mask_reader import NMask
from models.model import Model

"""
A mother class for basic n-gram model
Better prediction time:
> structure: tree = {index of word_A: [sub-tree, count], index of word_B: [sub-tree, count],...}
Less memory consumption:
> structure: dict = {(start index, end index): count, ...}
"""


class NGM(Model):
	def __init__(self, pre_words, post_words):
		Model.__init__(self, pre_words, post_words)

	def train(self, filename):
		with open(filename, "r", encoding="utf-8") as file:
			print(f">>> Reading {filename}")
			data = file.read()
		reader = NMask(data, self.pre_words, self.post_words)
		tenth = int(len(data)/10)
		currtenth = tenth
		while reader.e < len(data):
			if reader.e > currtenth:
				print(".", end="", flush=True)
				currtenth += tenth
			self.add_ngram(reader)
			reader.next_token()
		# We must add the last N-Gram
		self.add_ngram(reader)
		# Normalize the struct
		print()
		print("Done")

	def normalize(self):
		raise NotImplementedError("normalize not implemented")

	def add_ngram(self, data):
		raise NotImplementedError("add n-gram not implemented")

	def _predict(self, pre_word, post_word):
		"""
		Return every options predicted by the model with confidence values
		:param pre_word: the list of words before the word to predict
		:param post_word: the list of words after the word to predict
		:return: list of choices with confidence value [("wordA", confidence), ("wordB", confidence),...]
		"""
		raise NotImplementedError("_predict not implemented")

	def predict(self, pre_words, post_words, max_pred):
		options = self._predict(pre_words, post_words)
		# Sort and return the predictions by confidence
		options.sort(key=lambda x: x[1], reverse=True)
		# Limit the number of predictions as specified by the user
		if max_pred < len(options):
			options = options[:max_pred]
		return options
