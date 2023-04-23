python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install \
    datasets \
    ipywidgets \
    jupyterlab \
    lckr-jupyterlab-variableinspector \
    matplotlib \
    nltk \
    pandas \
    pyarrow \
    requests \
    scikit-learn \
    tensorboard \
    torch-lr-finder \
    tqdm
