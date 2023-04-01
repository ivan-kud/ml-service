import time

from fastapi import Request
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
LABELS = ['negative', 'neutral', 'positive']
MAX_TEXT_LENGTH = 300


class InputError(Exception):
    pass


def _timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_return = func(*args, **kwargs)
        end_time = time.time()
        return func_return, (end_time - start_time)
    return wrapper


def preprocess_text(text: str) -> str:
    # Check text length
    if len(text.strip()) < 1:
        raise InputError('Write a review please.')
    if len(text.strip()) > MAX_TEXT_LENGTH:
        text = text.strip()[:MAX_TEXT_LENGTH]

    # Replace usernames and links by placeholders
    token_list = []
    for token in text.split(' '):
        token = '@user' if token.startswith('@') and len(token) > 1 else token
        token = 'http' if token.startswith('http') else token
        token_list.append(token)

    return ' '.join(token_list)


@_timer
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    return tokenizer


@_timer
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    return model


@_timer
def predict(model, model_input) -> tuple[float, str]:
    model.eval()
    with torch.no_grad():
        logits = model(**model_input).logits

    probabilities = torch.nn.Softmax(dim=1)(logits)
    proba = probabilities.max().item()
    class_id = probabilities.argmax().item()
    label = LABELS[class_id]

    return proba, label


def get_response(text: str) -> dict[str, str | Request]:
    try:
        preprocessed_text = preprocess_text(text)
        tokenizer, token_load_time = load_tokenizer()
        model_input = tokenizer(preprocessed_text, return_tensors='pt')
        print(type(model_input))
        model, model_load_time = load_model()
        (proba, label), inference_time = predict(model, model_input)
    except InputError as err:
        return {'output1': str(err)}
    except Exception as err:
        err_msg = type(err).__name__ + ': ' + str(err)
        print(f'File "{__name__}",', err_msg)
        return {'output1': err_msg}

    # Form the info string
    info = ('Model load time: '
            + f'{int((token_load_time + model_load_time) * 1000.0)} ms. '
            + f'Inference time: {int(inference_time * 1000.0)} ms.')

    return {
        'text': text,
        'output1': 'Label: ' + label,
        'output2': 'Confidence: ' + f'{100*proba:.2f} %',
        'info': info,
    }
