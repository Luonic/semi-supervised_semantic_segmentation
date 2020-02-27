# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import glob
import logging
import time
import timeit
from pathlib import Path

import numpy as np
import io

import psycopg2
from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists, NoSuchKey

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import traceback
import cv2
from multiprocessing import JoinableQueue, Process

from albumentations import ReplayCompose, LongestMaxSize, PadIfNeeded, ToFloat

STOP = 'STOP'


def delete_image_from_db(cursor, image_id):
    cursor.execute(f"""DELETE FROM images WHERE images.image_id = '{image_id}'""")


def get_images_to_predict(cursor, model_name):
    cursor.execute(f"""SELECT * FROM (
SELECT images.image_id, images.image_name FROM images LEFT JOIN segmentation_confidences 
ON images.image_id = segmentation_confidences.image_id GROUP BY images.image_id
EXCEPT SELECT images.image_id, images.image_name FROM images LEFT JOIN segmentation_confidences 
ON images.image_id = segmentation_confidences.image_id
WHERE segmentation_confidences.model_name IS NOT DISTINCT FROM '{model_name}') t
ORDER BY RANDOM() LIMIT 500;""")
    image_ids_and_names = cursor.fetchmany(500)
    return image_ids_and_names


def download_object(minio_client, image_name):
    data = minio_client.get_object('images', image_name)
    image_bytes = io.BytesIO()
    for d in data.stream(32 * 1024):
        image_bytes.write(d)
    image_bytes.seek(0)
    return image_bytes


def download_and_preprocess_worker(output_queue):
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

    minio_client = Minio(S3_ENDPOINT,
                         access_key=S3_ACCESS_KEY,
                         secret_key=S3_SECRET_KEY,
                         secure=False)

    conn = psycopg2.connect(dbname=DB_DATABASE_NAME, user=DB_USERNAME, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    conn.autocommit = True

    while True:
        image_ids_and_names = get_images_to_predict(cursor, MODEL_NAME)

        for image_id, image_name in image_ids_and_names:
            try:
                image_bytes = download_object(minio_client, image_name)
                image_bytes_np = np.array(image_bytes.getbuffer(), dtype=np.uint8)
                image = cv2.imdecode(image_bytes_np, cv2.IMREAD_COLOR)

                if image is None:
                    print(f'Found video file {image_name}')
                    try:
                        minio_client.remove_object('images', image_name)
                    except ResponseError as err:
                        print(err)

                    delete_image_from_db(cursor, image_id)
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = augmentations(image=image)['image']
                image = np.transpose(image, axes=(2, 0, 1))

                image_batch_np = np.expand_dims(image, axis=0)
                output_queue.put((image_id, image_name, image_batch_np), block=True)
            except Exception as e:
                traceback.print_exc()

        else:
            time.sleep(5)


def predict_worker(input_queue, output_queue):
    MODEL_PATH = os.getenv('MODEL_PATH')
    MODEL_DEVICE = os.getenv('DEVICE', 'cpu')

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    # build model
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    model.to(MODEL_DEVICE)

    # for image_id, image_name, image_batch_np in iter(input_queue, STOP):
    with torch.no_grad():
        while True:
            image_id, image_name, image_batch_np = input_queue.get(block=True)
            image_batch = torch.from_numpy(image_batch_np).to(MODEL_DEVICE)

            prediction = model(image_batch)
            prediction = torch.sigmoid(prediction)
            binary_mask = (prediction > 0.5).to(prediction)
            confidence = torch.nn.functional.binary_cross_entropy(prediction, binary_mask, reduction='mean')

            confidence = confidence.cpu().item()
            print('image_name:', image_name, 'confidence:', confidence)
            output_queue.put((image_id, confidence), block=True)
            input_queue.task_done()


def load_worker(input_queue):
    DB_HOST = os.getenv('DB_HOST', '')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_DATABASE_NAME = os.getenv('DB_DATABASE_NAME', '')
    DB_USERNAME = os.getenv('DB_USERNAME', '')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')

    MODEL_NAME = os.getenv('MODEL_NAME')

    conn = psycopg2.connect(dbname=DB_DATABASE_NAME, user=DB_USERNAME, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    conn.autocommit = True

    # for image_id, confidence in iter(input_queue, STOP):
    while True:
        image_id, confidence = input_queue.get(block=True)
        try:
            cursor.execute(f"""INSERT INTO segmentation_confidences
                                               VALUES ('{str(image_id)}', '{str(confidence)}', '{str(MODEL_NAME)}')""")
        except NoSuchKey as e:
            delete_image_from_db(cursor, image_id)
            continue
        except Exception:
            traceback.print_exc()
        input_queue.task_done()


def main():
    NUM_EXTRACT_WORKERS = int(os.getenv('NUM_EXTRACT_WORKERS'))
    NUM_LOAD_WORKERS = int(os.getenv('NUM_LOAD_WORKERS'))

    images_to_predict_queue = JoinableQueue(maxsize=100)
    confidences_to_load_queue = JoinableQueue(maxsize=100)

    download_and_preprocess_workers = []
    for i in range(NUM_EXTRACT_WORKERS):
        p = Process(target=download_and_preprocess_worker, args=(images_to_predict_queue,))
        p.start()
        download_and_preprocess_workers.append(p)

    predict_workers = []
    p = Process(target=predict_worker, args=(images_to_predict_queue, confidences_to_load_queue))
    p.start()
    predict_workers.append(p)

    load_workers = []
    for i in range(NUM_LOAD_WORKERS):
        p = Process(target=load_worker, args=(confidences_to_load_queue,))
        p.start()
        load_workers.append(p)

    for worker in load_workers:
        worker.join()


# def main():
#
#     S3_ENDPOINT = os.getenv('S3_ENDPOINT', '')
#     S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY', '')
#     S3_SECRET_KEY = os.getenv('S3_SECRET_KEY', '')
#
#     DB_HOST = os.getenv('DB_HOST', '')
#     DB_PORT = os.getenv('DB_PORT', '5432')
#     DB_DATABASE_NAME = os.getenv('DB_DATABASE_NAME', '')
#     DB_USERNAME = os.getenv('DB_USERNAME', '')
#     DB_PASSWORD = os.getenv('DB_PASSWORD', '')
#
#     MODEL_PATH = os.getenv('MODEL_PATH')
#     MODEL_NAME = os.getenv('MODEL_NAME')
#     MODEL_DEVICE = os.getenv('DEVICE', 'cpu')
#     NUM_EXTRACT_WORKERS = os.getenv('NUM_EXTRACT_WORKERS')
#     NUM_LOAD_WORKERS = os.getenv('NUM_LOAD_WORKERS')
#
#     minio_client = Minio(S3_ENDPOINT,
#                         access_key=S3_ACCESS_KEY,
#                         secret_key=S3_SECRET_KEY,
#                         secure=False)
#
#     conn = psycopg2.connect(dbname=DB_DATABASE_NAME, user=DB_USERNAME, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
#     cursor = conn.cursor()
#     conn.autocommit = True
#
#     with torch.no_grad():
#
#         cudnn.benchmark = True
#         cudnn.deterministic = False
#         cudnn.enabled = True
#
#         # build model
#         model = torch.jit.load(MODEL_PATH)
#         model.to(MODEL_DEVICE)
#
#         augmentations = ReplayCompose([
#             LongestMaxSize(max_size=1024, always_apply=True),
#             PadIfNeeded(min_height=1024, min_width=1024, always_apply=True),
#             ToFloat()])
#
#         while True:
#             image_ids_and_names = get_images_to_predict(cursor)
#
#             for image_id, image_name in image_ids_and_names:
#                 try:
#                     data = minio_client.get_object('images', image_name)
#                     image_bytes = io.BytesIO()
#                     for d in data.stream(32 * 1024):
#                         image_bytes.write(d)
#                     image_bytes.seek(0)
#                     image_bytes = np.array(image_bytes.getbuffer(), dtype=np.uint8)
#                     image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
#
#                     if image is None:
#                         print(f'Found video file {image_name}')
#                         try:
#                             minio_client.remove_object('images', image_name)
#                         except ResponseError as err:
#                             print(err)
#
#                         delete_image_from_db(cursor, image_id)
#                         continue
#
#                     image = augmentations(image=image)['image']
#                     image = np.transpose(image, axes=(2, 0, 1))
#
#                     image_batch = np.expand_dims(image, axis=0)
#                     image_batch = torch.from_numpy(image_batch).to(MODEL_DEVICE)
#
#                     prediction = model(image_batch)
#                     prediction = torch.sigmoid(prediction)
#                     binary_mask = (prediction > 0.5).to(prediction)
#                     confidence = torch.nn.functional.binary_cross_entropy(prediction, binary_mask, reduction='mean')
#
#                     confidence = confidence.cpu().item()
#                     print('image_name:', image_name, 'confidence:', confidence)
#
#                     cursor.execute(f"""INSERT INTO segmentation_confidences
#                                    VALUES ('{str(image_id)}', '{str(confidence)}', '{str(MODEL_NAME)}')""")
#
#                 except NoSuchKey as e:
#                     delete_image_from_db(cursor, image_id)
#                     continue
#                 except Exception:
#                     traceback.print_exc()


if __name__ == '__main__':
    main()
