import io
import os
import traceback
import glob

import cv2
import numpy as np
import psycopg2
from minio import Minio
from minio.error import NoSuchKey
from operator import itemgetter
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from albumentations import ReplayCompose, LongestMaxSize, PadIfNeeded, ToFloat
MODEL_PATH = 'runs/14_hrnet-w48_coco-no-blank-pretrain/model.ts'
MODEL_DEVICE = 'cuda:0'
# MODEL_DEVICE = 'cpu'
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

num_clusters = 10

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


def download_image_from_s3_as_bytesio(minioClient, image_name):
    data = minioClient.get_object('images', image_name)
    image_bytes = io.BytesIO()
    for d in data.stream(32 * 1024):
        image_bytes.write(d)
    image_bytes.seek(0)
    return image_bytes

image_paths = []
features = []
entropies = []

print('predicting embeddings')
for (image_name,) in tqdm(image_names):
    try:
        print(image_name)
        image_bytes = download_image_from_s3_as_bytesio(minioClient, image_name)
        image_bytes_array = np.array(image_bytes.getbuffer(), dtype=np.uint8)
        image = cv2.imdecode(image_bytes_array, cv2.IMREAD_COLOR)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentations(image=image)['image']
        image = np.transpose(image, axes=(2, 0, 1))
        image_batch_np = np.expand_dims(image, axis=0)
        with torch.no_grad():
            image = torch.from_numpy(image_batch_np).to(MODEL_DEVICE)
            image_features, pred_logits = model(image)
            probs = torch.softmax(pred_logits[-1], dim=1)
            entropy = -(probs * torch.log(probs)).sum(dim=1).mean().item()
            for i in range(len(image_features)):
                image_features[i] = image_features[i].mean(dim=(0, 2, 3))
                # image_features[i] = torch.nn.functional.max_pool2d(image_features[i], kernel_size=4, stride=4).view((-1))
            image_features = torch.cat([image_features[-1]], dim=0).cpu().numpy()
            # image_features = torch.cat(image_features, dim=0).cpu().numpy()

        image_paths.append(image_name)
        features.append(image_features)
        entropies.append(entropy)

            # if IMAGE_SAVE_DIR != '':
            #
            #     if image_name not in existing_images:
            #         os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
            #         with open(os.path.join(IMAGE_SAVE_DIR.strip(), image_name), 'wb') as image_file:
            #             image_file.write(image_bytes.read())
    except NoSuchKey as e:
        continue
    except Exception as e:
        traceback.print_exc()

print('Clustering...')
kmeans = KMeans(n_clusters=num_clusters, tol=0.000001, n_jobs=8, max_iter=300).fit(np.array(features))
print('kmeans iters done', kmeans)
# kmeans = AgglomerativeClustering(n_clusters=num_clusters).fit(np.array(features))

print('Predicting clusters...')
clusters_list = [[] for i in range(num_clusters)]
for (image_name, ), image_features, entropy in tqdm(zip(image_names, features, entropies)):
    cluster_idx = kmeans.predict(image_features.reshape(1, -1))[0]
    clusters_list[cluster_idx].append((image_name, entropy))

print('Sorting...')
for cluster in tqdm(clusters_list):
    cluster.sort(key=itemgetter(1), reverse=True)

print('Writing result...')
for cluster_idx, cluster in tqdm(enumerate(clusters_list)):
    for image_name, entropy in cluster[:5]:
        bytes_array = download_image_from_s3_as_bytesio(minioClient, image_name)

        dst_path = os.path.join(IMAGE_SAVE_DIR, str(cluster_idx), os.path.basename(image_name))
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        print(entropy, dst_path)
        with open(dst_path, 'wb') as f:
            f.write(bytes_array.read())


print('Finished.')
