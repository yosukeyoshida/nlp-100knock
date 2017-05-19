from gensim import corpora, matutils

class Preprocessor():
    data_file = 'sentiment.txt'
    dic_file = 'dic.txt'

    @classmethod
    def create_dictionary(cls):
        docs, _ = cls.load_data()
        all_words = [[Util.normalize_word(word) for word in doc.split(' ') if not Util.is_invalid_word(word)] for doc in docs]

        dictionary = corpora.Dictionary(all_words)
        dictionary.filter_extremes(no_below=10)
        dictionary.save_as_text(cls.dic_file)
        return dictionary

    @classmethod
    def load_dictionary(cls):
        return corpora.Dictionary.load_from_text(cls.dic_file)

    @classmethod
    def load_data(cls):
        with open(cls.data_file) as f:
            samples = [line.rstrip().split('\t') for line in f]
            label = [sample[0] for sample in samples]
            docs = [sample[1] for sample in samples]
        return [docs, label]

    @staticmethod
    def split_normalized_words(docs):
        return [[Util.normalize_word(word) for word in doc.split(' ')] for doc in docs]

    @classmethod
    def create_dense(cls, docs):
        dictionary = cls.load_dictionary()
        normalized_docs = cls.split_normalized_words(docs)
        corpus = [dictionary.doc2bow(doc) for doc in normalized_docs]
        dense = [list(matutils.corpus2dense([line], num_terms=len(dictionary)).T[0]) for line in corpus]
        return dense

    @classmethod
    def run(cls):
        cls.create_dictionary()
        docs, y = Preprocessor.load_data()
        X = Preprocessor.create_dense(docs)
        return [X, y]

