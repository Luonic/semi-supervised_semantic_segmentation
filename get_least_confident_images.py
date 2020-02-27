import io
import os
import traceback
import glob

import cv2
import numpy as np
import psycopg2
from minio import Minio
from minio.error import NoSuchKey

import torch
from albumentations import ReplayCompose, LongestMaxSize, PadIfNeeded, ToFloat
MODEL_PATH = 'runs/25_focal-loss-a0.5g0.25x2_staged_backbone_finetuning/model.ts'
MODEL_DEVICE = 'cuda:0'
model = torch.jit.load(MODEL_PATH)
model.to(MODEL_DEVICE)
augmentations = ReplayCompose([
        LongestMaxSize(max_size=1024, always_apply=True),
        PadIfNeeded(min_height=1024, min_width=1024, always_apply=True),
        ToFloat()])


S3_ENDPOINT = os.getenv('S3_ENDPOINT', '')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY', '')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY', '')

DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_DATABASE_NAME = os.getenv('DB_DATABASE_NAME', '')
DB_USERNAME = os.getenv('DB_USERNAME', '')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

MODEL_NAME = os.getenv('MODEL_NAME')

IMAGE_SAVE_DIR = os.getenv('IMAGE_SAVE_DIR', '')
if MODEL_NAME == '':
    raise ValueError('Environment variable MODEL_NAME not set.')

minioClient = Minio(S3_ENDPOINT,
                    access_key=S3_ACCESS_KEY,
                    secret_key=S3_SECRET_KEY,
                    secure=False)

conn = psycopg2.connect(dbname=DB_DATABASE_NAME, user=DB_USERNAME, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()
conn.autocommit = True

# cursor.execute(f"""SELECT images.image_name FROM images, segmentation_confidences
#                    WHERE segmentation_confidences.image_id = images.image_id
#                    AND segmentation_confidences.model_name = '{MODEL_NAME}' ORDER BY confidence DESC;""")
cursor.execute(f"""SELECT images.image_name FROM images ORDER BY RANDOM();""")
image_names = cursor.fetchmany(10000)

existing_images = [os.path.basename(img_path) for img_path in glob.glob(
    os.path.join(os.path.dirname(IMAGE_SAVE_DIR), '**', '*.jpg'), recursive=True)]

for (image_name,) in image_names[:1000]:
    try:
        data = minioClient.get_object('images', image_name)
        image_bytes = io.BytesIO()
        for d in data.stream(32 * 1024):
            image_bytes.write(d)
        image_bytes.seek(0)
        image_bytes_array = np.array(image_bytes.getbuffer(), dtype=np.uint8)
        image = cv2.imdecode(image_bytes_array, cv2.IMREAD_COLOR)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_image = image = augmentations(image=image)['image']
        image = np.transpose(image, axes=(2, 0, 1))
        image_batch_np = np.expand_dims(image, axis=0)
        with torch.no_grad():
            image = torch.from_numpy(image_batch_np).to(MODEL_DEVICE)
            pred_map = model(image)
            pred_map = torch.sigmoid(pred_map)
            mask = pred_map[0, 1:]
            binary_mask = (mask > 0.5).to(mask)
            binary_mask = np.transpose(binary_mask.cpu().detach().numpy(), axes=(1, 2, 0))
            mask = np.transpose(mask.cpu().detach().numpy(), axes=(1, 2, 0))
        red_tensor = np.array([[[1, 0, 0]]], dtype=np.float32)
        visualization_mask = binary_mask * red_tensor

        resize_ratio = 800 / max(orig_image.shape)
        image = cv2.cvtColor(cv2.resize(orig_image, None, fx=resize_ratio, fy=resize_ratio), cv2.COLOR_RGB2BGR)
        visualization_mask = cv2.resize(visualization_mask, (image.shape[1], image.shape[0]))
        binary_mask = np.expand_dims(cv2.resize(binary_mask, (image.shape[1], image.shape[0])), axis=2)
        image = image * (1 - binary_mask) + (image * (0.5 * binary_mask) + visualization_mask * (0.5 * binary_mask))
        mask = cv2.resize(mask, None, fx=resize_ratio, fy=resize_ratio)

        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        key = cv2.waitKey(0)
        if key == 13:
            if IMAGE_SAVE_DIR != '':

                if image_name not in existing_images:
                    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
                    with open(os.path.join(IMAGE_SAVE_DIR.strip(), image_name), 'wb') as image_file:
                        image_file.write(image_bytes.read())
    except NoSuchKey as e:
        continue
    except Exception as e:
        traceback.print_exc()
