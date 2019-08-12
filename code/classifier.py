from src.utils import Utils
from src.vectorGenerator import VectorGenerator
import pandas as pd
import datetime


class Classifier:

    def __init__(self, base_path, stock_ticker, time_frame, stock_price, document_vector):
        self.pickle_path = base_path + '/pickle/'
        self.stock_ticker = stock_ticker
        self.time_frame = time_frame
        self.stock_price = stock_price
        self.doc_vector = document_vector
        self.train_data = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def label_documents(self):

        for i in range((self.stock_price.Date.count())):
            today = self.stock_price['Date'].values[i]
            cur_time = self.stock_price['Time'].values[i]
            if cur_time.hour == 9:
                yesterday = today - pd.Timedelta('24:00:00')
                evening = datetime.time(17, 30, 0)
                yesterday_evening = self.doc_vector[(self.doc_vector['Date'] == yesterday)
                                                    & (self.doc_vector['Time'] >= evening)]
                today_morning = self.doc_vector[(self.doc_vector['Date'] == today)
                                                & (self.doc_vector['Time'] < cur_time)]
                documents = pd.concat([yesterday_evening, today_morning], ignore_index=True)
                if len(documents) != 0:
                    start_index = yesterday_evening[0].index[0]
                    end_index = today_morning[0].index[-1]
                    combined_document_vector = VectorGenerator(documents, self.stock_ticker,
                                                               self.pickle_path).combine_document_vectors(start_index,
                                                                                                          end_index)
            else:
                interval_start_time = datetime.time(cur_time.hour - (int(self.time_frame) // 60), cur_time.minute)
                documents = self.doc_vector[(self.doc_vector['Date'] == today)
                                            & (self.doc_vector['Time'] >= interval_start_time)
                                            & (self.doc_vector['Time'] < cur_time)]
                if len(documents) != 0:
                    start_index = documents[0].index[0]
                    end_index = documents[0].index[-1]
                    combined_document_vector = VectorGenerator(documents, self.stock_ticker,
                                                               self.pickle_path).combine_document_vectors(start_index,
                                                                                                          end_index)
            self.train_data.loc[i] = combined_document_vector
            combined_document_vector = None

        self.train_data['Label'] = self.stock_price['Direction'].values
        self.train_data.to_csv(self.pickle_path + self.stock_ticker + '_traindata_' + self.time_frame + '.csv', sep=',',
                               index=False)

