from .ngram_tree import NG_Tree


class NG_Tree_Backoff(NG_Tree):
	def __init__(self, pre_words, post_words):
		NG_Tree.__init__(self, pre_words, post_words)

	def prior(self, pre_words):
		node = self.root
		try:
			for word in pre_words:
				node = node[word][0]
			return node
		except KeyError:
			return {}

	def retro(self, prior, post_words):
		nodes = {}
		for key, (child, count) in prior.items():
			try:
				testnode = child
				for word in post_words:
					testnode = testnode[word][0]
				nodes[key] = (child, count)
			except KeyError:
				pass
		return nodes

	def _predict(self, pre_words, post_words):
		node = self.prior(pre_words)
		if len(node) > 0:
			node = self.retro(node, post_words)
		if len(node) == 0:
			if len(pre_words) > len(post_words):
				return self._predict(pre_words[1:], post_words)
			else:
				return self._predict(pre_words, post_words[:-1])
		return [(word, count) for word, (child, count) in node.items()]
