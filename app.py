#!/usr/bin/env python3

#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# app.py
# @author Seonho Oh
# @description DeepStream 기반 우범여행자 추적 프로그램
# @created 2023-05-23 15:00:00
#
"""DeepStream 기반 우범여행자 추적 프로그램."""

from __future__ import annotations

import argparse
import configparser
import ctypes
import math
import os
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from utils import message
from utils.config import *
from utils.fps import FPS
from utils.general import *

import gi  # noqa E401, E402, isort: skip

gi.require_version("Gst", "1.0")  # isort: skip
gi.require_version("GstRtspServer", "1.0")  # isort: skip
from gi.repository import GLib, Gst, GstRtspServer  # noqa E401, E402, isort: skip

import pyds  # noqa E401, E402, isort: skip

fps_streams: dict[int, FPS] = {}

CAMERA_IDS: list[int] = []


def caps_src_pad_buffer_probe(unused_pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: Any) -> Gst.PadProbeReturn:
    """사람 검출 결과와 재식별 특징벡터, 얼굴 검출 결과를 전송.

    Args:
        unused_pad (Gst.Pad): Unused
        info (Gst.PadProbeInfo): 현재 Probe의 정보
        unused_data (gpointer): Unused

    Returns:
        Gst.PadProbeReturn: Gst.PadProbeReturn.OK
    """
    prep_queue, meta_queue, thumbnail_queue = user_data
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get Gst.Buffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    reidFeatures: np.ndarray = None
    l_user_meta = batch_meta.batch_user_meta_list
    while l_user_meta is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
        except StopIteration:
            break

        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_BATCH_REID_META:
            reidTensor = pyds.NvDsReidTensorBatch.cast(user_meta.user_meta_data)
            reidFeatures = reidTensor.get_features()

        try:
            l_user_meta = l_user_meta.next
        except StopIteration:
            break

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        ntp_timestamp = int(
            datetime.fromtimestamp(round(frame_meta.ntp_timestamp / Gst.SECOND, 3)).strftime("%Y%m%d%H%M%S%f")[:-3]
        )

        persons = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            text_params = obj_meta.text_params
            text_params.font_params.font_name = "NanumGothic"
            text_params.display_text = f"{obj_meta.obj_label} {obj_meta.object_id & 0xFFFFFFFF}"

            if obj_meta.object_id != UNTRACKED_OBJECT_ID:
                if obj_meta.obj_label == "person":
                    rect_params = obj_meta.rect_params
                    box = np.asarray(
                        [
                            rect_params.left,
                            rect_params.top,
                            rect_params.width,
                            rect_params.height,
                        ]
                    )
                    box[2:] += box[:2]

                    feature = None
                    if reidFeatures is not None:
                        l_user_meta = obj_meta.obj_user_meta_list
                        while l_user_meta is not None:
                            try:
                                user_meta = pyds.NvDsUserMeta.cast(l_user_meta.data)
                            except StopIteration:
                                break

                            if (
                                user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_OBJ_REID_META
                                and user_meta.user_meta_data
                            ):
                                reidInd = ctypes.cast(
                                    pyds.get_ptr(user_meta.user_meta_data),
                                    ctypes.POINTER(ctypes.c_int32),
                                ).contents.value
                                if reidInd >= 0 and reidInd < reidTensor.numFilled:
                                    feature = reidFeatures[reidInd, :]

                            try:
                                l_user_meta = l_user_meta.next
                            except StopIteration:
                                break

                    persons.append(
                        OrderedDict(
                            {
                                "object_id": obj_meta.object_id & 0xFFFFFFFF,
                                "object_type": "person",
                                "bbox": box.tolist(),
                                "feature": feature if feature is None else feature.tolist(),
                            }
                        )
                    )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        rgb = None
        if persons:
            if rgb is None:
                arr = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                rgba = np.array(arr, copy=True, order="C")
                rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

            for i, person in enumerate(persons):
                if person["feature"] is not None:
                    x1, y1 = map(int, person["bbox"][:2])
                    x2, y2 = map(lambda x: int(math.ceil(x)), person["bbox"][2:])

                    thumbnail_queue.put(
                        OrderedDict(
                            {
                                "ntp_timestamp": ntp_timestamp,
                                "source_id": CAMERA_IDS[frame_meta.pad_index],
                                "camera_id": CAMERA_IDS[frame_meta.pad_index],
                                "object_id": person["object_id"],
                                "object_type": person["object_type"],
                                "image": rgb[y1 : y2 + 1, x1 : x2 + 1, :].copy(),
                            }
                        )
                    )

            meta_queue.put(
                {
                    "ntp_timestamp": ntp_timestamp,
                    "source_id": CAMERA_IDS[frame_meta.pad_index],
                    "camera_id": CAMERA_IDS[frame_meta.pad_index],
                    "objects": persons,
                }
            )

        fps_streams[frame_meta.pad_index]()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def send_meta(in_queue: Queue[Any | None], collection_name: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger.info("Start send_meta process")

    import pymongo

    collection: pymongo.collection.Collection = None

    if ENABLE_DB:
        try:
            conn = pymongo.MongoClient(f"mongodb://{DB_USER}:{DB_PASSWORD}@{DB_IP}:{DB_PORT}", tz_aware=True)
            db = conn[DB_NAME]
            collection = db[collection_name]
        except pymongo.errors.ConnectionFailure as e:
            print(f"Could not connect to server: {e}")

    if ENABLE_KAFKA:  # auto batching
        import msgpack
        from kafka import KafkaProducer

        producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=msgpack.dumps, linger_ms=40)

    while True:
        try:
            item = in_queue.get_nowait()

            if item is None:
                break

            if "objects" in item:  # person
                if ENABLE_KAFKA:
                    producer.send(KAFKA_TOPIC_REID, item)

                ntp_timestamp = item["ntp_timestamp"]
                source_id = item["source_id"]
                camera_id = item["camera_id"]

                objs = []
                for obj in item["objects"]:
                    objs.append(
                        OrderedDict(
                            {
                                "ntp_timestamp": ntp_timestamp,
                                "source_id": source_id,
                                "camera_id": camera_id,
                                **obj,
                            }
                        )
                    )

                try:
                    collection.insert_many(objs)
                except (pymongo.errors.BulkWriteError, OverflowError) as e:
                    logger.error(e)
                except Exception as e:
                    logger.error(e)
            else:
                producer.send(KAFKA_TOPIC_FACE, item)
                try:
                    collection.insert_one(item)
                except (pymongo.errors.BulkWriteError, OverflowError) as e:
                    logger.error(e)
                except Exception as e:
                    logger.error(e)
        except Empty:
            time.sleep(0.01)

    if ENABLE_DB:
        conn.close()

    logger.info("Stop send_meta process")


def main(args: Any) -> None:
    global CAMERA_IDS

    cv2.setNumThreads(0)

    IS_TEGRA = is_aarch64()

    torch.backends.cudnn.benchmark = True

    description = "DeepStream 기반 우범여행자 추적 프로그램\n"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--uris",
        nargs="+",
        type=str,
        help="List of URIs. --uris uri1 uri2 ... uriN",
        required=True,
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="List of camera IDs. --ids id1 id2 ... idN",
        required=True,
    )

    parser.add_argument(
        "--sink",
        type=int,
        help="Sink type (1: FakeSink, 2: EGLSink, 3: FileSink, 4: RTSP). Default sink is 2.",
        default=2,
    )

    parser.add_argument(
        "--sync-always",
        action="store_true",
        help="Sync always",
    )

    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    num_sources = len(args.uris)

    CAMERA_IDS = args.ids

    manager = mp.Manager()

    meta_queue = manager.Queue(MAX_QUEUE_SIZE)
    thumbnail_queue = manager.Queue(MAX_QUEUE_SIZE)

    meta_proc = mp.Process(
        target=send_meta,
        args=(
            meta_queue,
            COLLECTION_META,
        ),
    )
    meta_proc.start()

    thumbnail_proc = mp.Process(
        target=message.send_thumbnail,
        args=(
            thumbnail_queue,
            COLLECTION_THUMBNAIL,
        ),
    )
    thumbnail_proc.start()

    # waiting for initialization
    logger.info("Waiting for subprocess initialization...")

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    logger.info("Creating Pipeline")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        logger.error("Unable to create Pipeline")

    logger.info("Creating streamux")

    # Create nvstreammux instance to form batches from a source.
    streammux = make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")
    pipeline.add(streammux)

    for i in range(num_sources):
        logger.info(f"Creating source_bin {i}")
        uri_name = args.uris[i]

        if uri_name.find("rtsp://") == 0:
            is_live = True
        elif uri_name.find("file://") == 0:
            file_name = uri_name[7:]
            file_name = os.path.abspath(file_name)
            uri_name = f"file://{file_name}"

        source_bin = create_source_bin(i, uri_name, rate=None)
        if not source_bin:
            logger.error("Unable to create source bin")
        pipeline.add(source_bin)

        padname = f"sink_{i}"
        link_source_to_mux(source_bin, streammux, padname, "video")

        fps_streams[i] = FPS(i)

    # if not USE_NEW_NVSTREAMMUX:
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    streammux.set_property("batch-size", num_sources)
    streammux.set_property("attach-sys-ts", 1)

    if is_live:
        logger.info("At least one of the sources is live")
        streammux.set_property("live-source", 1)

    tracker = make_elm_or_print_err("nvtracker", "tracker")

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read(TRACKER_CONFIG_FILE)
    config.sections()

    for key in config["tracker"]:
        if key in TRACKER_CONFIG_INT_KEYS:
            tracker.set_property(key, config.getint("tracker", key))
        elif key in ["ll-lib-file", "ll-config-file"]:
            tracker.set_property(key, config.get("tracker", key))

    tracker.set_property("display-tracking-id", 1)

    nvvidconv0 = make_elm_or_print_err("nvvideoconvert", "converter0", "Nvvidconv")

    # convert color space
    caps0 = Gst.ElementFactory.make("capsfilter", "filter0")
    caps0.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"),
    )

    if args.sink > 1:
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")

        tiler_rows = int(math.ceil(math.sqrt(num_sources)))
        # tiler_rows = int(math.sqrt(num_sources))
        tiler_columns = int(math.ceil((1.0 * num_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("height", TILED_OUTPUT_HEIGHT)

        # Create OSD to draw on the converted RGBA buffer
        nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "OSD (nvosd)")

        nvosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvosd.set_property("display-text", OSD_DISPLAY_TEXT)
        nvosd.set_property("display-bbox", OSD_DISPLAY_BBOX)

        if IS_TEGRA:
            transform = make_elm_or_print_err("nvegltransform", "nvegl-transform", "EGL Transform")

    if args.sink == 1:
        sink = make_elm_or_print_err("fakesink", "fakesink", "FakeSink")
    elif args.sink == 2:
        sink = make_elm_or_print_err("nveglglessink", "nvvideo-renderer", "EGLSink")
        # sink = make_elm_or_print_err("nvvideoconvert", "nvvidconv-fps")
        # fpssink = make_elm_or_print_err("fpsdisplaysink", "FPSdisplaysink")
    elif args.sink == 3 or args.sink == 4:
        nvvidconv1 = make_elm_or_print_err("nvvideoconvert", "convertor_postosd", "nvvidconv_postosd")

        # Create a caps filter
        caps1 = Gst.ElementFactory.make("capsfilter", "filter1")
        caps1.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=I420" if SUPPORT_HW_ENC else "video/x-raw, format=I420"
            ),
        )

        codec = CODEC.lower()

        # Make the encoder
        encoder = make_elm_or_print_err(
            f"nvv4l2{codec}enc" if SUPPORT_HW_ENC else f"x{codec[1:]}enc",
            "encoder",
            f"{CODEC} Encoder",
        )

        encoder.set_property("bitrate", BITRATE if SUPPORT_HW_ENC else BITRATE // 1000)

        if not SUPPORT_HW_ENC:
            encoder.set_property("tune", "zerolatency")
            encoder.set_property("key-int-max", 30)
        else:
            encoder.set_property("tuning-info-id", 3)  # ultralowlatency
            # encoder.set_property("iframeinterval", 15)  # default 30

        if args.sink == 3:
            sink = make_elm_or_print_err("filesink", "filesink", "FileSink")
            sink.set_property(
                "location",
                "out.mp4" if is_live or num_sources > 1 else f"{args.uris[0][7:-4]}_out.mp4",
            )

            codeparser = make_elm_or_print_err(f"{codec}parse", f"{codec}parser", "Code Parser")
            container = make_elm_or_print_err("qtmux", "qtmux", "Container")
        else:
            # Make the payload-encode video into RTP packets
            rtppay = make_elm_or_print_err(f"rtp{codec}pay", "rtppay", f"{CODEC} rtppay")
            rtppay.set_property("config-interval", 1)

            sink = make_elm_or_print_err("udpsink", "udpsink")

            sink.set_property("host", "0.0.0.0")
            sink.set_property("port", UDPSINK_PORT_NUM)
    else:
        logger.error("Invalid argument for sink")
        sys.exit(1)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so args.frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        # comment out to avoid "gst-stream-error-quark: memory type configured and i/p buffer mismatch ip_surf 2 muxer 3 (1)"
        # streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv0.set_property("nvbuf-memory-type", mem_type)

        if args.sink > 1:
            tiler.set_property("nvbuf-memory-type", mem_type)
            if args.sink > 2:
                nvvidconv1.set_property("nvbuf-memory-type", mem_type)

    logger.info("Adding elements to Pipeline")

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    print("daadasd")
    print(pgie)

    pgie_person = create_gie("primary", "group", "./configs/config_group.txt")

    elems = [pgie_person, tracker, nvvidconv0, caps0]

    if args.sink > 1:
        elems += [tiler, nvosd]
        if args.sink == 2 and IS_TEGRA:
            elems.append(transform)
        elif args.sink == 3:
            elems += [nvvidconv1, caps1, encoder, codeparser, container]
        elif args.sink == 4:  # RTSP
            elems += [nvvidconv1, caps1, encoder, rtppay]

    elems.append(sink)

    for i, elem in enumerate(elems):
        pipeline.add(elem)

    N = len(elems)
    if args.sink == 2 and IS_TEGRA:
        N = N - 1

    queue = []
    for i in range(N):
        q = Gst.ElementFactory.make("queue", f"queue{i}")
        queue.append(q)
        pipeline.add(q)

    logger.info("Linking elements in the Pipeline")
    streammux.link(queue[0])
    for i in range(N):
        queue[i].link(elems[i])
        if queue[i] is not queue[-1]:
            elems[i].link(queue[i + 1])

    if args.sink == 2 and IS_TEGRA:
        transform.link(sink)
        # sink.link(fpssink)
    else:
        queue[-1].link(sink)

    sink.set_property("sync", 0)

    if args.sync_always:
        logger.info("Set sync 1")
        sink.set_property("sync", 1)

    # create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    if args.sink == 4:
        # Start streaming
        server = GstRtspServer.RTSPServer.new()
        server.props.service = str(RTSP_PORT_NUM)
        server.attach(None)

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            f"( udpsrc name=pay0 port={UDPSINK_PORT_NUM} buffer-size=524288 "
            f'caps="application/x-rtp, media=video, clock-rate=90000, '
            f'encoding-name=(string){CODEC}, payload=96 " )'
        )
        factory.set_shared(True)
        server.get_mount_points().add_factory("/analytics", factory)

        print(f"\n *** DeepStream: Launched RTSP Streaming at " f"rtsp://localhost:{RTSP_PORT_NUM}/analytics ***\n\n")

    caps0_src_pad = caps0.get_static_pad("src")
    if not caps0_src_pad:
        logger.error(" Unable to get sink pad of nvosd")
    caps0_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        caps_src_pad_buffer_probe,
        (None, meta_queue, thumbnail_queue),
    )

    # start play back and listen to events
    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except BaseException:
        logger.info("Sending an EOS event to the pipeline")
        pipeline.send_event(Gst.Event.new_eos())

        logger.info("Waiting for the EOS message on the bus")
        bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS)

    pipeline.set_state(Gst.State.NULL)

    logger.info("Terminating meta proc")
    meta_queue.put(None)
    meta_proc.join()

    logger.info("Terminating thumbnail proc")
    thumbnail_queue.put(None)
    thumbnail_proc.join()

    print("Done")


if __name__ == "__main__":
    main(sys.argv)
