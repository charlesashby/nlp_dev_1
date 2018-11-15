from models.ngram.ngram_model import NGM
import itertools
from pympler import asizeof
from sys import intern


class NG_Regular(NGM):
	def __init__(self, pre_words, post_words):
		NGM.__init__(self, pre_words, post_words)
		self.tab = {}
		self.uni = {}

	def add_ngram(self, reader):
		j = len(reader.words)
		while j > 0:
			if j == 1:
				unigram = reader.words[0]
				if unigram in self.uni:
					self.uni[unigram] += 1
				else:
					self.uni[unigram] = 1
			else:
				ngram = " ".join(itertools.islice(reader.words, 0, j))
				if ngram in self.tab:
					self.tab[ngram] += 1
				else:
					self.tab[ngram] = 1
			j -= 1

	def getsize(self):
		return asizeof.asizeof(self.tab)

	def _predict(self, pre_word, post_word):
		preds = []
		if len(pre_word) > 0:
			pregram = " ".join(pre_word) + " "
		else:
			pregram = ""
		if len(post_word) > 0:
			postgram = " " + " ".join(post_word)
		else:
			postgram = ""
		pre = len(pre_word)
		n = pre + len(post_word) + 1
		for unigram in self.uni:
			ngram = pregram + unigram + postgram
			if ngram in self.tab:
				preds.append((unigram, self.tab[ngram]))

		return preds
