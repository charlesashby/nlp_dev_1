class UnknownWordError(Exception):
	def __init__(self, message):
		Exception.__init__(self, message)
		self.message = message


class Model:
	def __init__(self, pre_words, post_words):
		# The number of words before the prediction that the model will work with
		self.pre_words = pre_words
		# number of words after the prediction
		self.post_words = post_words

	def train(self, training_file):
		"""
		Trains the model to prepare it for predictions
		:param training_file: The path to the training file to be used
		"""
		raise NotImplementedError("Training function not implemented")

	def predict(self, pre_words, post_words, max_pred):
		"""
		A function used to evaluate the model after its training
		:param pre_words: A list of words preceding the prediction (in order)
		:param post_words: A list of words after the prediction
		:param max_pred: the maximum size of the prediction list to return
		:return: [(prediction_A, confidence), (prediction_B, confidence),...] maximum size: max_pred
		"""
		raise NotImplementedError("Prediction function not implemented")

	def getsize(self):
		"""
		Functions that returns the size in bytes of the model
		:return: size in bytes of the model
		"""
	@staticmethod
	def display_predictions(predictions, pre_words, post_words):
		"""
		display the predictions withing the context and their confidence level
		:param pre_words: A list of words preceding the prediction (in order)
		:param post_words: A list of words after the prediction
		:param predictions: [(prediction_A, confidence), (prediction_B, confidence),...]
		"""
		pre = " ".join(pre_words)
		post = " ".join(post_words)
		print("Top Predictions:")
		maxsize = len(max(predictions, key=lambda x: len(x[0]))[0])
		for prediction in predictions:
			spaces = 1 + maxsize-len(prediction[0])
			print(f"{pre} _{prediction[0]}_ {post}" + spaces*" " + f"->  {prediction[1]}")
