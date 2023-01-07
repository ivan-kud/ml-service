from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import utils


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
def predict_digit(request: Request, img: str = Form()):
    # Preprocess image to use it as model input
    image_tensor = utils.preprocess_digit(img)

    # Predict
    model = utils.digit_models['convl4fconn1']
    proba, label = utils.predict_digit(model, image_tensor)

    return templates.TemplateResponse('digit.html', {
        'request': request,
        'output1': f'Label: {label}',
        'output2': f'Confidence: {100*proba:.2f} %',
        'image': img,
    })


@app.get('/person', response_class=HTMLResponse)
def read_person(request: Request):
    return templates.TemplateResponse('person.html', {'request': request})
