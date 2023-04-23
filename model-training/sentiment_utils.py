import os
import pickle

import requests
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             HashingVectorizer)
import tqdm


nltk.download('stopwords')


class DatasetError(Exception):
    pass


class Tokenizer:
    def __init__(self):
        self.stop_words = nltk.corpus.stopwords.words('english')
        self.tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()

    def __call__(self, text: str, return_str: bool = False) -> list[str] | str:
        # To lower case
        tokens = text.lower()

        # Tokenize
        tokens = self.tokenizer.tokenize(tokens)

        tokens_temp = []
        for token in tokens:
            # Remove stop words
            if token in self.stop_words:
                continue

            # Replace usernames and links by placeholders
            token = '@user' if token.startswith('@') and len(token) > 1 else token
            token = 'http' if token.startswith('http') else token

            # Stemming
            token = self.stemmer.stem(token, to_lowercase=False)

            tokens_temp.append(token)
        tokens = tokens_temp

        # Add a word if len is zero
        if len(tokens) == 0:
            tokens = ['word']

        return ' '.join(tokens) if return_str else tokens


def preprocess_text(text: str) -> str:
    # Replace usernames and links by placeholders
    tokens = text.split(' ')
    for token in text.split(' '):
        token = '@user' if token.startswith('@') and len(token) > 1 else token
        token = 'http' if token.startswith('http') else token
        tokens.append(token)

    return ' '.join(tokens)


def get_bow_and_tfifd(dataset, n_features, svd_components, file, saving=True):
    """Returns BOW vectorizer from file if it exists.
    Otherwise, this function initializes and fits vectorizer.
    """
    # Load vectorizer if it already exists
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            output = pickle.load(f)
    else:
        # Initialize tokenizer
        tokenizer = Tokenizer()

        # Initialize vectorizers
        bow_vectorizer = CountVectorizer(lowercase=False,
                                         tokenizer=tokenizer,
                                         max_features=n_features)
        tfidf_vectorizer = TfidfTransformer()
        hashing_vectorizer = HashingVectorizer(lowercase=False,
                                               tokenizer=tokenizer,
                                               n_features=n_features)

        # Initialize SVD-truncated vectorizers
        svd_bow_vectorizer = TruncatedSVD(n_components=svd_components)
        svd_tfidf_vectorizer = TruncatedSVD(n_components=svd_components)
        svd_hashing_vectorizer = TruncatedSVD(n_components=svd_components)

        # Fit vectorizers and transform train data
        x_train_bow = bow_vectorizer.fit_transform(dataset['train']['text'])
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_bow)
        x_train_hashing = hashing_vectorizer.fit_transform(dataset['train']['text'])

        # Fit SVD-truncated vectorizers and transform train data
        x_train_svd_bow = svd_bow_vectorizer.fit_transform(x_train_bow)
        x_train_svd_tfidf = svd_tfidf_vectorizer.fit_transform(x_train_tfidf)
        x_train_svd_hashing = svd_hashing_vectorizer.fit_transform(x_train_hashing)

        # Transform validation and test data
        x_valid_bow = bow_vectorizer.transform(dataset['validation']['text'])
        x_valid_tfidf = tfidf_vectorizer.transform(x_valid_bow)
        x_valid_hashing = hashing_vectorizer.transform(dataset['validation']['text'])
        x_test_bow = bow_vectorizer.transform(dataset['test']['text'])
        x_test_tfidf = tfidf_vectorizer.transform(x_test_bow)
        x_test_hashing = hashing_vectorizer.transform(dataset['test']['text'])

        # Transform validation and test data for SVD-truncated vectorizers
        x_valid_svd_bow = svd_bow_vectorizer.transform(x_valid_bow)
        x_valid_svd_tfidf = svd_tfidf_vectorizer.transform(x_valid_tfidf)
        x_valid_svd_hashing = svd_hashing_vectorizer.transform(x_valid_hashing)
        x_test_svd_bow = svd_bow_vectorizer.transform(x_test_bow)
        x_test_svd_tfidf = svd_tfidf_vectorizer.transform(x_test_tfidf)
        x_test_svd_hashing = svd_hashing_vectorizer.transform(x_test_hashing)

        # Form output
        output = (
            bow_vectorizer, x_train_bow, x_valid_bow, x_test_bow,
            tfidf_vectorizer, x_train_tfidf, x_valid_tfidf, x_test_tfidf,
            hashing_vectorizer, x_train_hashing, x_valid_hashing, x_test_hashing,
            svd_bow_vectorizer, x_train_svd_bow, x_valid_svd_bow, x_test_svd_bow,
            svd_tfidf_vectorizer, x_train_svd_tfidf, x_valid_svd_tfidf, x_test_svd_tfidf,
            svd_hashing_vectorizer, x_train_svd_hashing, x_valid_svd_hashing, x_test_svd_hashing,
        )

        # Save vectorizers and transformed data
        if saving:
            with open(file, 'wb') as f:
                pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    return output


def dataset_query(api_url):
    response = requests.request('GET', api_url)
    return response.json()


def download_dataset(dataset_name, dataset_conf, dataset_path):
    query = 'is-valid'
    api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
    api_response = dataset_query(api_url)
    if api_response['valid']:
        query = 'parquet'
        api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
        api_response = dataset_query(api_url)
        for config in api_response['parquet_files']:
            if config['config'] == dataset_conf:
                response = requests.get(config['url'], stream=True)
                file_name = config['url'].split('/')[-1]
                file_path = dataset_path + file_name
                if not os.path.isfile(file_path):
                    with open(file_path, 'wb') as handle:
                        for data in tqdm(response.iter_content()):
                            handle.write(data)
    else:
        raise DatasetError(f'Dataset "{dataset_name}" is not valid.')
