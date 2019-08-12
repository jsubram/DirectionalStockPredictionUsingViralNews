from datetime import datetime, date
import pytz
import pickle
import pandas as pd


class Utils:
    @staticmethod
    def save_as_pickle(data, basepath, fileName):
        fileObject = open(basepath + fileName + ".p", 'wb')
        pickle.dump(data, fileObject)
        fileObject.close()
        print('pickled ', fileName)

    @staticmethod
    def read_pickle(basepath, fileName):
        fileObject = open(basepath + fileName, 'rb')
        df = pickle.load(fileObject)
        return df
