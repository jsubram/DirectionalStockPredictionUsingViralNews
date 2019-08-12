import sys
from src.stockProcessor import StockProcessor
from src.models import Models
from src.newsProcessor import NewsProcessor
from src.classifier import Classifier

if __name__ == '__main__':
    print('load Stock chart data')
    base_file_path = sys.argv[1]
    stockCharts = StockProcessor(base_file_path)
    print('Input stock price interval in minutes - 5, 15, 30, 60, 240, 1440')
    time_interval = input()
    amazon_stock_prices, apple_stock_prices = stockCharts.loadDataForInterval(time_interval)
     For training only
     document_vectors_amazon, document_vectors_apple = NewsProcessor(base_file_path).loadNewsArticles()
     # AMAZON
     classify = Classifier(base_file_path, 'amazon', time_interval, amazon_stock_prices, document_vectors_amazon)
     classify.label_documents()
     # APPLE
     classify = Classifier(base_file_path, 'apple', time_interval, apple_stock_prices, document_vectors_apple)
     classify.label_documents()
    amazon_model = Models(base_file_path, 'amazon', amazon_stock_prices, time_interval)
    amazon_model.naive_bayes_classifier()
    amazon_model.SVM_classifier()
    amazon_model.DT_classifier()
    amazon_model.SVM_poly_classifier()
    amazon_model.Logistic_Regression11_classifier()
    amazon_model.Logistic_Regression12_classifier()
    amazon_model.KNN_classifier()
    amazon_model.SGDC_classifier()
    amazon_model.accounting_factor()

    apple_model = Models(base_file_path, 'apple', apple_stock_prices, time_interval)
    apple_model.naive_bayes_classifier()
    apple_model.SVM_classifier()
    apple_model.DT_classifier()
    apple_model.SVM_poly_classifier()
    apple_model.Logistic_Regression11_classifier()
    apple_model.Logistic_Regression12_classifier()
    apple_model.KNN_classifier()
    apple_model.SGDC_classifier()

    print()

    # ACCOUNTING FACTOR
