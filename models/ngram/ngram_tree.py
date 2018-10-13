from models.model import Model
from models.mask_reader import NMask


class NG_Tree(Model):
	@staticmethod
	def _display(node, tabs):
		text = ""
		nexttabs = tabs+"│\t"
		endtabs = tabs + "\t"
		for i, (word, (child, count)) in enumerate(node.items()):
			text += f"{tabs}│\n"
			if i < len(node) - 1:
				text += f"{tabs}├── {word} ({count})\n"
				text += NG_Tree._display(child, nexttabs)
			else:
				text += f"{tabs}└── {word} ({count})\n"
				text += NG_Tree._display(child, endtabs)
		return text

	def __init__(self, pre_word, post_word):
		Model.__init__(self, pre_word, post_word)
		# root = {"word_A": [sub-tree, count], "word_B": [sub-tree, count],...}
		self.root = {}

	def display(self):
		print(NG_Tree._display(self.root, ""))

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

	def add_ngram(self, reader):
		i = 0
		node = self.root
		# This ensure that only 1 copy of the same word is stored (pointers to the copy are used as key)
		while i < reader.n:
			word = reader.words[i]
			if word in node:
				node[word][1] += 1
			else:
				node[word] = [{}, 1]
			node = node[word][0]
			i += 1

	def _normalize(self, node):
		# Compute the sum of all the nodes
		total = sum(i for _, i in node.values())
		for child in node.values():
			child[1] /= total
			self._normalize(child[0])

	def normalize(self):
		"""
		To be call after the training, transform the counting to relative frequencies
		"""
		# Compute the sum of all the nodes
		total = sum(i for _, i in self.root.values())
		for child in self.root.values():
			child[1] /= total
			self._normalize(child[0])


	@staticmethod
	def merge_options(main_opt, sub_opt, punish):
		for word, freq in sub_opt.items():
			if word in main_opt:
				main_opt[word] *= freq
			else:
				main_opt[word] = freq*punish

	@staticmethod
	def node_to_options(node):
		# options = {"wordA": count, "wordB": count, ...}
		options = {}
		for word, child in node.items():
			options[word] = child[1]
		return options

	def _predict(self, pre_words, post_words):
		raise NotImplementedError("predict not implemented")

	def predict(self, pre_words, post_words, max_pred):
		options = self._predict(pre_words, post_words)
		# Sort and return the predictions by confidence
		options.sort(key=lambda x: x[1], reverse=True)
		# Limit the number of predictions as specified by the user
		if max_pred < len(options):
			options = options[:max_pred]
		return options
