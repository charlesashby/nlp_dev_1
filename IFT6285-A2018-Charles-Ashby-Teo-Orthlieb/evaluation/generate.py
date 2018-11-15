def generate(model, start):
	pre = len(start)
	while start[-1] != ".":
		preds = model.predict(start[max(len(start)-model.pre_words, 0):], [], 5)
		if len(start) < 7:
			# The sentence is still short, we avoid the dots
			token = preds[0][0]
			if token == ".":
				token = preds[1][0]
			start.append(token)
		elif len(start) < 12:
			# The sentence is reasonably sized, we take the best prediction
			start.append(preds[0][0])
		else:
			# The sentence is long, we look for a dot
			for pred in preds:
				if pred[0] == ".":
					start.append(".")
					break
			else:
				start.append(preds[0][0])
		if len(start) > 50:
			# It's getting out of hand, we must stop
			start.append("[...]")

	# Capitalize the beginning of the sentence
	start[0] = "(" + start[0][0].upper() + start[0][1:]
	start[pre-1] += ")"
	return " ".join(start)