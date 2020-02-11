import io
import os
import traceback
import glob

import cv2
import numpy as np
import psycopg2
from minio import Minio
from minio.error import NoSuchKey

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

cursor.execute(f"""SELECT images.image_name FROM images, segmentation_confidences 
                   WHERE segmentation_confidences.image_id = images.image_id 
                   AND segmentation_confidences.model_name = '{MODEL_NAME}' ORDER BY confidence DESC;""")
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
        cv2.imshow('image', image)
        cv2.waitKey(1)
        if IMAGE_SAVE_DIR != '':

            if image_name not in existing_images:
                os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
                with open(os.path.join(IMAGE_SAVE_DIR.strip(), image_name), 'wb') as image_file:
                    image_file.write(image_bytes.read())
    except NoSuchKey as e:
        continue
    except Exception as e:
        traceback.print_exc()
