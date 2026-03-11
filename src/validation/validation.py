import json
import io
import uuid

from .exif_check import check, CheckException

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pathlib import Path
from google.cloud import storage


# ================================================
BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = BASE_DIR / 'best_model.pt'
BASE_MODEL = BASE_DIR / 'yolov8n-cls.pt'
# ================================================

router = APIRouter()
MAX_IMAGES_PER_INFRASTRUCTURE = 3

_bucket = None
_model = None
_device = None

class ResponseBody(BaseModel):
    message: str


def get_bucket():
    global _bucket
    if _bucket is None:
        client = storage.Client()
        _bucket = client.bucket('sportlink-b061c.firebasestorage.app')
    return _bucket


def get_model_and_device():
    global _model, _device
    if _model is None:
        import torch
        from torch import nn
        from ultralytics import YOLO

        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(WEIGHTS_PATH, map_location=_device)
        classifier_key = next((k for k in state_dict.keys() if k.endswith('linear.weight')), None)
        if classifier_key is None:
            raise KeyError('Could not find classifier linear.weight in saved state_dict.')

        num_classes = state_dict[classifier_key].shape[0]
        yolo = YOLO(BASE_MODEL)
        model = yolo.model
        in_features = model.model[-1].linear.in_features
        model.model[-1].linear = nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.to(_device)
        model.eval()
        _model = model

    return _model, _device


@router.post("")
async def predict(
    file: UploadFile = File(...), 
    exif: str = Form(...),
    infrastructure: str = Form(...)
) -> ResponseBody:
    from .inference import infer

    contents = await file.read()
    imageData = io.BytesIO(contents)

    exif_data = json.loads(exif)
    infrastructure_data = json.loads(infrastructure)
    model, device = get_model_and_device()

    conf, label = infer(
        imageData=imageData, 
        model=model,
        device=device
    )
    print(f'inference: label={label}, confidence={conf:.4f}')

    try:
        check(
            confidence=conf,
            label=label,
            exif=exif_data,
            infrastructure=infrastructure_data
        )
    except CheckException as e:
        raise HTTPException(status_code=400, detail=e.error.value)

    try: 
        save(
            file_bytes=contents,
            infra_id=infrastructure_data['infra_id'],
            date=exif_data['date_taken']
        )
    except ValueError as e:
        if str(e).startswith('max_images_reached_for_infrastructure:'):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'failed_to_save_image: {e}')

    return {'message': '✅ Image uploaded!'}


def save(file_bytes, infra_id, date):
    bucket = get_bucket()
    prefix = f"infrastructures/{infra_id}/"
    existing_count = sum(1 for _ in bucket.list_blobs(prefix=prefix, max_results=MAX_IMAGES_PER_INFRASTRUCTURE + 1))
    if existing_count >= MAX_IMAGES_PER_INFRASTRUCTURE:
        raise ValueError(f"max_images_reached_for_infrastructure:{infra_id}")

    unique_id = str(uuid.uuid4())
    path = f"{prefix}{unique_id}_{date}.jpg"

    blob = bucket.blob(path)

    blob.upload_from_file(
        io.BytesIO(file_bytes),
        content_type='image/jpeg'
    )
