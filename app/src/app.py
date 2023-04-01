from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import digit
from . import instance
from . import sentiment


app = FastAPI()

templates = Jinja2Templates(directory='templates')
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/', response_class=HTMLResponse)
def read_home(request: Request):
    return templates.TemplateResponse('pages/index.html', {'request': request})


@app.get('/digit', response_class=HTMLResponse)
def read_digit(request: Request):
    return templates.TemplateResponse('pages/digit.html', {'request': request})


@app.post('/digit', response_class=HTMLResponse)
def predict_digit(request: Request,
                  model_name: digit.ModelName = Form(),
                  image: str = Form()):
    response = digit.get_response(model_name, image)
    response.update({'request': request})
    return templates.TemplateResponse('pages/digit.html', response)


@app.get('/instance', response_class=HTMLResponse)
def read_instance(request: Request):
    return templates.TemplateResponse('pages/instance.html',
                                      {'request': request})


@app.post('/instance', response_class=HTMLResponse)
def predict_instance(request: Request,
                     file: UploadFile):
    response = instance.get_response(file)
    response.update({'request': request})
    return templates.TemplateResponse('pages/instance.html', response)


@app.get('/sentiment', response_class=HTMLResponse)
def read_digit(request: Request):
    return templates.TemplateResponse('pages/sentiment.html',
                                      {'request': request})


@app.post('/sentiment', response_class=HTMLResponse)
def predict_digit(request: Request,
                  text: str = Form()):
    response = sentiment.get_response(text)
    response.update({'request': request})
    return templates.TemplateResponse('pages/sentiment.html', response)
