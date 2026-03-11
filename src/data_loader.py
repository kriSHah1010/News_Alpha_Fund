import pandas as pd


class DataLoader:

    def __init__(self, news_path, reddit_path, price_path):
        self.news_path = news_path
        self.reddit_path = reddit_path
        self.price_path = price_path

    def load_news(self):
        news = pd.read_csv(self.news_path)
        news['Date'] = pd.to_datetime(news['Date'])
        return news

    def load_reddit(self):
        reddit = pd.read_csv(self.reddit_path)
        reddit['Date'] = pd.to_datetime(reddit['Date'])
        return reddit

    def load_prices(self):
        prices = pd.read_csv(self.price_path)
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices.sort_values('Date', inplace=True)
        return prices