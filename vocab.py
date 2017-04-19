
class Vocabulary(object):
    ''' A two-way mapping between words in our vocabulary and indexes. This
        means that one can look up a word by index or an index by word. '''
    def __init__(self, words):
        self.word_to_index_map = {}
        self.index_to_word_map = {}

        word_set = set(words)

        for index, word in enumerate(word_set):
            self.word_to_index_map[word] = index
            self.index_to_word_map[index] = word

        self.length = len(word_set)


    def __len__(self):
        return self.length


    def __getitem__(self, key):
        try:
            return self.index_to_word_map[key]
        except KeyError:
            return self.word_to_index_map[key]
