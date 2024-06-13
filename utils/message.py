from __future__ import annotations

import base64
import signal
import time
from queue import Empty
from typing import Any

import cv2
import pymongo

from .config import *
from .general import logger

META_INTERVAL = 15
META_BATCH_TIMEOUT = 0.1
META_MAX_BATCH_SIZE = 256


def send_meta(in_queue: Any, collection: pymongo.collection.Collection) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger.info("Start send_meta process")

    condition = True
    while condition:
        start = time.time()
        batch = []

        while True:
            try:
                item = in_queue.get_nowait()

                if item is None:
                    condition = False
                else:
                    batch.append(item)
                end = time.time()

                if (end - start) > META_BATCH_TIMEOUT or len(batch) == META_MAX_BATCH_SIZE:
                    if batch:
                        collection.insert_many(batch)
                    break
            except Empty:
                if batch:
                    collection.insert_many(batch)
                else:
                    time.sleep(0.01)
                break

    logger.info("Stop send_meta process")


def send_thumbnail(in_queue: Any, collection_name: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger.info("Start send_thumbnail process")

    collection: pymongo.collection.Collection = None

    if ENABLE_DB:
        try:
            conn = pymongo.MongoClient(f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_IP}:{DB_PORT}", tz_aware=True)
            db = conn[DB_NAME]
            collection = db[collection_name]
        except pymongo.errors.ConnectionFailure as e:
            print(f"Could not connect to server: {e}")

    condition = True
    while condition:
        start = time.time()
        batch = []

        while True:
            try:
                item = in_queue.get_nowait()

                if item is None:
                    condition = False
                    break

                if item["image"].size > 0:
                    thumbnail = item["image"]

                    h, w = thumbnail.shape[:2]

                    # resize thumbnail if too large
                    if h > 384:
                        thumbnail = cv2.resize(thumbnail, (int(w * 384 / h), 384))

                    ret, thumbnail = cv2.imencode(
                        ".jpg",
                        thumbnail[..., ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )
                    item["image_b64"] = base64.b64encode(thumbnail).decode("utf-8")
                    del item["image"]
                    batch.append(item)
            except Empty:
                time.sleep(0.01)

            end = time.time()

            if (end - start) > META_BATCH_TIMEOUT or len(batch) == META_MAX_BATCH_SIZE:
                break

        if batch:
            collection.insert_many(batch)

    if ENABLE_DB:
        conn.close()

    logger.info("Stop send_thumbnail process")
