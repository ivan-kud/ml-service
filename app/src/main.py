from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import digit


app = FastAPI()

templates = Jinja2Templates(directory='app/templates')
app.mount('/static', StaticFiles(directory='app/static'), name='static')


@app.get('/', response_class=HTMLResponse)
def read_home(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/digit', response_class=HTMLResponse)
def read_digit(request: Request):
    return templates.TemplateResponse('digit.html', {'request': request})


@app.post('/digit', response_class=HTMLResponse)
def predict_digit(request: Request,
                  model_name: digit.ModelName = Form(),
                  image: str = Form()):
    data = digit.get_response_data(model_name, image)
    return templates.TemplateResponse('digit.html', {'request': request, 'data': data})


@app.get('/person', response_class=HTMLResponse)
def read_person(request: Request):
    return templates.TemplateResponse('person.html', {'request': request})
