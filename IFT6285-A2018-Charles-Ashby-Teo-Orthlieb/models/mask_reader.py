from collections import deque
from sys import intern


class NMask:
	def __init__(self, data, pre_words, post_words, banned=(',', ';'), reset=()):
		self.data = data.lower()
		self.pre_words = pre_words
		# The n consecutive words
		self.n = pre_words + post_words + 1
		# A deque of n consecutive words
		self.words = deque()
		# Reader's position in text
		self.s = 0
		self.e = 0
		# The list of char that we will ignore thanks to this abstraction
		self.banned = banned
		self.reset = reset
		self.init()

	def init(self):
		# Initialize self.words so that it contains the n first words
		# While there's not n words in self.words
		while len(self.words) < self.n:
			self.add_token()

	def add_token(self):
		self.s = self.e
		# Move forward until we hit a white char
		while self.e < len(self.data) and not self.data[self.e].isspace():
			self.e += 1
		word = intern(self.data[self.s:self.e].lower())
		# Skip the space
		self.e += 1
		# Check if the word is banned
		if word in self.banned:
			# Skip the banned word
			self.add_token()
		else:
			# Add the non-banned word to our deque
			self.words.append(word)

	def next_token(self):
		self.words.popleft()
		self.add_token()
