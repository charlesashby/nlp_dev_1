from .ngram_tree import NG_Tree


class NG_Tree_KN(NG_Tree):
	def __init__(self, pre_words, post_words):
		NG_Tree.__init__(self, pre_words, post_words)

	def prior(self, pre_words):
		delta = 0.2

		node = self.root
		try:
			for word in pre_words:
				node = node[word][0]
		except KeyError:
			node = self.root
		return node

	def retro(self, prior, post_words):
		# TODO: faire ça apres
		return prior
