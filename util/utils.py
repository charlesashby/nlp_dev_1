import re, unicodedata


def tokenize_sentence(sentence):
    tokens = sentence.split(' ')
    tokens_clean = []
    for i, _ in enumerate(tokens):
        if tokens[i] == '<unk':
            tokens_clean.append('{} {}'.format(tokens[i], tokens[i + 1]))
        elif 'w="' == tokens[i][:3]:
            pass
        else:
            tokens_clean.append(tokens[i])
    return tokens_clean


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_line(line):
    cleaned_tokens = []
    for token in line.split(' '):
        try:
            if is_number(token):
                cleaned_tokens.append('<NUMBER>')
            elif '.\n' in token and token != '.\n':
                cleaned_tokens.append(token.replace('.\n', '').lower())
            else:
                cleaned_tokens.append(strip_accents(token).lower())
        except UnicodeDecodeError:
            pass
    cleaned_line = ' '.join([t for t in cleaned_tokens])
    return cleaned_line


def get_token_dict_pos(token_dict, token):
    if '<unk w="' in token:
        pos = -1
        return pos
    elif token == '<s>':
        pos = -3
        return pos
    try:
        pos = token_dict.index(token)
    except ValueError:
        # print('not found: {}'.format(token))
        pos = -2
    return pos


def structure_file(file, suf=3, pre=5, clean=False):
    structured_unks = []
    lines = open(file, 'r').readlines()

    # clean lines...

    for j, line in enumerate(lines):
        if clean:
            cleaned_line = clean_line(line)
        else:
            cleaned_line = line
        tokens = tokenize_sentence(cleaned_line)
        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            if '<unk w="' in token:
                end = min(n_tokens, i + suf)
                start = max(0, i - pre)
                history = tokens[start:end + 1]
                if len(history) == pre + suf + 1:
                    structured_unk = history
                elif i + suf >= n_tokens:
                    structured_unk = ['<s>' for _ in range(pre + suf + 1)]
                    structured_unk[:len(history)] = history
                elif i - pre < 0:
                    structured_unk = ['<s>' for _ in range(pre + suf + 1)]
                    structured_unk[-len(history):] = history
                else:
                    print('hello')
                    # return NotImplementedError
                structured_unk[pre] = '<UNK>'
                y = re.search('<unk w="(.*)"/>', token).group(1)
                structured_unks.append([structured_unk, y])
    return structured_unks
