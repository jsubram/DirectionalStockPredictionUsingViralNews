from src.utils import Utils
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec
import logging
import tempfile

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TEMP_FOLDER = tempfile.gettempdir()
# print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))


class VectorGenerator:
    def __init__(self, df, name, picklepath):
        self.df = df
        self.ticker_name = name
        self.taggedData = []
        self.pickle_path = picklepath

    def tokenize_text(self):
        for i in range(self.df.title.count()):
            text = self.df['title'][i] + ". " + self.df['text'][i]
            self.taggedData.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [i]))
            # TODO: change number of records that can be pickled
            #        try to comment lines 26-29 and see if it runs on whole dataset.
            # if i % 2000 == 0 and i != 0:
            #     Utils.save_as_pickle(self.taggedData, self.pickle_path, 'train_corpus_' + self.ticker_name + str(i))
            #     print('saved ', i, len(self.taggedData), ' ', self.df.title.count())
            #     self.taggedData = []
            # yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [i])
        Utils.save_as_pickle(self.taggedData, self.pickle_path, 'train_corpus_' + self.ticker_name + str(i))
        print('saved ', i, len(self.taggedData), ' ', self.df.title.count())

    def generate_model(self, fname):
        model = Doc2Vec(vector_size=20, window=4, min_count=4, workers=4, alpha= 0.025, min_alpha = 0.025)
        model.build_vocab(self.taggedData)
        model.train(self.taggedData, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(fname)

    def generate_doc_2_vec_representation(self):
        # TODO: if pickle exists load from there else tokenize text and save as pickle files
        self.tokenize_text()
        self.taggedData = Utils.read_pickle(self.pickle_path, 'train_corpus_' + self.ticker_name + '.p')
        # TODO: if pickle exists load from there else tokenize text and save as pickle files
        fname = get_tmpfile("my_doc2vec_model_" + self.ticker_name)
        self.generate_model(fname)
        model = Doc2Vec.load(fname)
        return model.docvecs.vectors_docs

    def combine_document_vectors(self, start_index, end_index):
        self.taggedData = Utils.read_pickle(self.pickle_path, 'train_corpus_' + self.ticker_name + '.p')
        self.taggedData = self.taggedData[start_index:end_index]
        tokens = []
        for i in self.taggedData:
            for word in i.words:
                if word not in tokens:
                    tokens.append(word)
        fname = get_tmpfile("my_doc2vec_model_" + self.ticker_name)
        model = Doc2Vec.load(fname)
        return model.infer_vector(tokens)
