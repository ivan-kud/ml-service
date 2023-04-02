import os

import requests
import nltk
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

        # Remove stop words
        tokens = [t for t in tokens if t not in self.stop_words]

        # Replace usernames and links by placeholders
        tokens = ['@user' if t.startswith('@') and len(t) > 1 else t for t in tokens]
        tokens = ['http' if t.startswith('http') else t for t in tokens]

        # Stemming
        tokens = [self.stemmer.stem(t, to_lowercase=False) for t in tokens]

        # Add a word if len is zero
        if len(tokens) == 0:
            tokens = ['word']

        return ' '.join(tokens) if return_str else tokens


def dataset_query(api_url):
    response = requests.request('GET', api_url)
    return response.json()


def download_dataset(dataset_name, dataset_path):
    query = 'is-valid'
    api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
    api_response = dataset_query(api_url)
    if api_response['valid']:
        query = 'parquet'
        api_url = f'https://datasets-server.huggingface.co/{query}?dataset={dataset_name}'
        api_response = dataset_query(api_url)
        for config in api_response['parquet_files']:
            if config['config'] == 'sentiment':
                response = requests.get(config['url'], stream=True)
                file_name = config['url'].split('/')[-1]
                file_path = dataset_path + file_name
                if not os.path.isfile(file_path):
                    with open(file_path, 'wb') as handle:
                        for data in tqdm(response.iter_content()):
                            handle.write(data)
    else:
        raise DatasetError(f'Dataset "{dataset_name}" is not valid.')
