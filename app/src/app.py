from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import digit


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
    response_data = digit.get_response_data(model_name, image)
    response_data.update({'request': request})
    return templates.TemplateResponse('pages/digit.html', response_data)


@app.get('/person', response_class=HTMLResponse)
def read_instance(request: Request):
    return templates.TemplateResponse('pages/person.html',
                                      {'request': request})
