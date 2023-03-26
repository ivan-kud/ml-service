import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
LABELS = ['negative', 'neutral', 'positive']


def preprocess_text(text: str) -> str:
    """Preprocess text (username and link placeholders)"""
    preprocessed_text = []
    for t in text.split(' '):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)


def get_response_data(text: str) -> dict:
    # Check the input
    if len(text.strip()) < 1:
        return {'output1': 'Write a review please.'}

    # Preprocessed text
    preprocessed_text = preprocess_text(text)
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')

    # Predict
    with torch.no_grad():
        logits = model(**encoded_input).logits
    probabilities = torch.nn.Softmax(dim=1)(logits)
    proba = probabilities.max().item()
    class_id = logits.argmax().item()
    label = LABELS[class_id]

    return {
        'text': text,
        'output1': 'Label: ' + label,
        'output2': 'Confidence: ' + f'{100*proba:.2f} %',
    }


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
