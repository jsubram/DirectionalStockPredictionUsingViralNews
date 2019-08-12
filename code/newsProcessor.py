import pandas as pd
import json
import os
import pickle

from src.utils import Utils
from src.vectorGenerator import VectorGenerator


class NewsProcessor:
    idx = 0

    def __init__(self, base_path):
        self.json_path = base_path + '/swm-project/news/'
        self.pickle_path = base_path + '/pickle/'
        self.df = pd.DataFrame(columns=['published', 'title', 'text'])
        self.amazon_df = pd.DataFrame(columns=['published', 'title', 'text'])
        self.apple_df = pd.DataFrame(columns=['published', 'title', 'text'])
        self.apple = ['AAPL', 'aapl', 'apple', 'Apple', 'APPLE']
        self.amazon = ['AMZN', 'amzn', 'amazon', 'Amazon', 'AMAZON']

    def load_news_articles_from_JSON(self):
        folders = [d for r, d, f in os.walk(self.json_path)][0]
        for folder in folders:
            json_files = [file_json for file_json in os.listdir(self.json_path + folder + '/') if
                          file_json.endswith('.json')]
            for file in json_files:
                file_name = self.json_path + folder + '/' + file
                data = json.load(open(file_name, encoding="utf8"))
                if data['language'] == 'english':
                    self.df.loc[NewsProcessor.idx] = [data['published'], data['title'], data['text']]
                    NewsProcessor.idx += 1
            print(folder)
            Utils.save_as_pickle(self.df, self.pickle_path, folder)

    def formatNewsDocuments(self):
        # standardize the news timezone to stock prices timezone that is converting UTC to EST time.
        self.df['published'] = pd.to_datetime(self.df['published'], utc=True)
        self.df['published'] = self.df['published'] - pd.Timedelta('04:00:00')
        self.df = self.df.sort_values(['published'])
        # Filtering out news from 2018 onwards
        self.df = self.df[self.df['published'] >= '2018-1-1 00:00:00']
        self.df = self.df.reset_index(drop=True)

    def filter_articles(self):
        idx_apple = 0
        idx_amazon = 0
        for i in range(len(self.df)):
            content = self.df['title'][i] + ". " + self.df['text'][i]
            for find_txt in self.apple:
                if find_txt in content:
                    self.apple_df.loc[idx_apple] = self.df.loc[i]
                    idx_apple += 1
                    break
            for find_txt in self.amazon:
                if find_txt in content:
                    self.amazon_df.loc[idx_amazon] = self.df.loc[i]
                    idx_amazon += 1
                    break
        # print('total count', self.df.count())
        # print('number of articles in amazon ', self.amazon_df.count())
        # print('\n number of articles in apple ', self.apple_df.count())
        Utils.save_as_pickle(self.amazon_df, self.pickle_path, 'amazon_news_df')
        Utils.save_as_pickle(self.apple_df, self.pickle_path, 'apple_news_df')

    def loadNewsArticles(self):
        # TODO: if pickle exists load from there else load from JSON
        self.load_news_articles_from_JSON()
        self.df = Utils.read_pickle(self.pickle_path, '2018_08.p')
        self.formatNewsDocuments()
        # TODO: if amazon and apple news have been separated out in 2 dfs and if the pickle file for the same exists,
        #       load from there, else call the filter method
        self.filter_articles()
        self.amazon_df = Utils.read_pickle(self.pickle_path, 'amazon_news_df.p')
        self.apple_df = Utils.read_pickle(self.pickle_path, 'apple_news_df.p')
        self.amazon_df = self.amazon_df.drop_duplicates(subset=['published', 'title'])
        self.amazon_df = self.amazon_df.drop_duplicates(subset=['published', 'title'])
        # generating document vectors for amazon news articles
        amazon_news = VectorGenerator(self.amazon_df, 'amazon', self.pickle_path)
        document_vectors_amazon = pd.DataFrame(amazon_news.generate_doc_2_vec_representation())
        # attaching the corresponding gmt time to the resulting document_vectors
        document_vectors_amazon['Date'] = self.amazon_df['published'].head(len(document_vectors_amazon)).dt.date
        document_vectors_amazon['Time'] = self.amazon_df['published'].head(len(document_vectors_amazon)).dt.time

        self.amazon_df = self.amazon_df.reset_index(drop=True)
        self.apple_df = self.apple_df.reset_index(drop=True)

        # generating document vectors for apple news articles
        apple_news = VectorGenerator(self.apple_df, 'apple', self.pickle_path)
        document_vectors_apple = pd.DataFrame(apple_news.generate_doc_2_vec_representation())
        # attaching the corresponding gmt time to the resulting document_vectors
        document_vectors_apple['Date'] = self.apple_df['published'].head(len(document_vectors_apple)).dt.date
        document_vectors_apple['Time'] = self.apple_df['published'].head(len(document_vectors_apple)).dt.time
        return document_vectors_amazon, document_vectors_apple
