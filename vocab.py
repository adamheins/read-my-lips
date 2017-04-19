
class Vocabulary(object):
    ''' A two-way mapping between words in our vocabulary and indexes. This
        means that one can look up a word by index or an index by word. '''
    def __init__(self, words):
        # Mapping of words to indices.
        self.word_to_index_map = {}

        # Mapping of indices to words.
        self.index_to_word_map = {}

        # Mapping of words to number of occurrences.
        self.occurrences_map = {}

        word_set = set(words)

        for index, word in enumerate(word_set):
            self.word_to_index_map[word] = index
            self.index_to_word_map[index] = word
            self.occurrences_map[word] = 0

        for word in words:
            self.occurrences_map[word] += 1

        self.total = len(words)
        self.length = len(word_set)


    def __len__(self):
        return self.length


    def __getitem__(self, key):
        try:
            return self.index_to_word_map[key]
        except KeyError:
            return self.word_to_index_map[key]


    def occurrences(self, word=None):
        if word:
            return self.occurrences_map[word]
        else:
            return self.occurrences_map
