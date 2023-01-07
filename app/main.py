from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from torchvision.models import resnet50, ResNet50_Weights

from app import utils


app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Initialize weights, transforms and model
# weights = ResNet50_Weights.DEFAULT
# preprocess_image = weights.transforms()
# model = resnet50(weights=weights)


@app.get('/', response_class=HTMLResponse)
def read_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/digit', response_class=HTMLResponse)
def read_digit(request: Request):
    return templates.TemplateResponse("digit.html", {"request": request})


@app.post('/digit', response_class=HTMLResponse)
def predict_digit(request: Request, img: str = Form()):
    # Preprocess image to use it as model input
    image_tensor = utils.preprocess_digit(img)

    # Predict
    model = utils.digit_models['convl4fconn1']
    proba, label = utils.predict_digit(model, image_tensor)

    return templates.TemplateResponse("digit.html", {"request": request,
                                                     'message': (proba, label),
                                                     'image': img})


# def preprocess_image(img_mx_array, model, thresh=0.5):
#     # Preprocess an image
#     img_mx_array, img_np_array = gluoncv.data.transforms.presets.rcnn.transform_test(img_mx_array)
#
#     # Inference
#     ids, scores, bboxes, masks = [x[0].asnumpy() for x in model(img_mx_array)]
#
#     # Filter by person id
#     person_id = model.classes.index('person')
#     scores = scores[ids[:, 0] == person_id, :]
#     bboxes = bboxes[ids[:, 0] == person_id, :]
#     masks = masks[ids[:, 0] == person_id, :, :]
#     ids = ids[ids[:, 0] == person_id, :]
#
#     # Paint segmentation mask on images directly
#     width, height = img_np_array.shape[1], img_np_array.shape[0]
#     masks, _ = gluoncv.utils.viz.expand_mask(masks, bboxes, (width, height), scores, thresh)
#     out_img_np_array = gluoncv.utils.viz.plot_mask(img_np_array, masks)
#
#     # Make a Figure and add it to canvas
#     dpi = 80
#     figsize = (out_img_np_array.shape[1] / float(dpi), out_img_np_array.shape[0] / float(dpi))
#     fig = Figure(figsize=figsize, dpi=dpi)
#     ax = fig.add_axes([0, 0, 1, 1])
#     ax.axis('off')
#
#     # Paint bboxes, scores and classes on images directly
#     ax = gluoncv.utils.viz.plot_bbox(out_img_np_array, bboxes, scores, ids, thresh,
#                                      model.classes, ax=ax, linewidth=2.0, fontsize=8)
#
#     fig.savefig('image.png')