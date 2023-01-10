# Run application locally
uvicorn app.src.main:app --reload

# Run jupyter-lab
jupyter-lab

# Run tensorboard
tensorboard --logdir=runs
