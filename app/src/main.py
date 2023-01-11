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
    # Preprocess image to use it as model input
    image_tensor = digit.preprocess_image(image)

    # Get a model and predict
    result = {}
    for name, model in digit.models.items():
        if model_name is digit.ModelName.all or model_name == name:
            result[name] = digit.predict(model, image_tensor)

    # Form result as strings
    if len(result) == 1:
        result_str_1 = f'{result[model_name][1]}'
        result_str_2 = f'{100*result[model_name][0]:.2f} %'
    else:
        result_str_1 = '; '.join([f'{name} - {value[1]}      '
                                  for name, value in result.items()])
        result_str_2 = '; '.join([f'{name} - {100*value[0]:.2f} %'
                                  for name, value in result.items()])

    return templates.TemplateResponse('digit.html', {
        'request': request,
        'model_name': model_name,
        'image': image,
        'output1': 'Label: ' + result_str_1,
        'output2': 'Confidence: ' + result_str_2,
    })


@app.get('/person', response_class=HTMLResponse)
def read_person(request: Request):
    return templates.TemplateResponse('person.html', {'request': request})
