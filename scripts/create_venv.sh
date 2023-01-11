python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install \
    tqdm \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    tensorboard \
    jupyterlab \
    ipywidgets \
    lckr-jupyterlab-variableinspector \
    torch-lr-finder