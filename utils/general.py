#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# general.py
# @author Seonho Oh
# @description https://github.com/NVIDIA-AI-IOT/deepstream_python_apps의 코드를 일부 발췌하고, 수정함
# @created 2023-05-31 14:30:00
#
# Original license as follows
#
################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""Define utility methods."""
from __future__ import annotations

import logging
import platform
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.distance import cdist

from .config import *

from .logger import SafeRotatingFileHandler  # noqa: E402, isort: skip

import gi  # noqa: E402, isort: skip

gi.require_version("Gst", "1.0")  # isort: skip
from gi.repository import Gst  # noqa: E402, isort: skip

import pyds  # noqa: E402, isort: skip


logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
logger.addHandler(sh)
fh = SafeRotatingFileHandler("logs/app.log", maxBytes=655350, backupCount=10)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s "))
logger.addHandler(fh)


UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


def is_aarch64() -> bool:
    """ARM64 환경 여부 검사."""
    return platform.uname()[4] == "aarch64"


def bus_call(unused_bus: Any, message: Any, loop: Any) -> bool:
    """Gstreamer pipeline bus message handler."""
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error(f"Error: {err}: {debug}")
        loop.quit()
    return True


def make_elm_or_print_err(factoryname: str, name: str, printedname: str = "", detail: str = "") -> Gst.Element:
    """Creates an element with Gst Element Factory make.

    Return the element  if successfully created, otherwise print to
    stderr and return None.
    """
    if not printedname:
        printedname = name

    logger.info(f"Creating {printedname}")
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        logger.error(f"Unable to create {printedname}")
        if detail:
            sys.stderr.write(detail)
    return elm


def cb_newpad(decodebin: Any, decoder_src_pad: Any, data: Any) -> None:
    logger.info("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    nbin = data

    # Need to check if the pad created by the decodebin is for audio and not
    # video.
    logger.info(f"gstname={gstname}")
    if gstname.find("audio/x-raw") != -1:
        index = nbin.name.split("-")[-1]
        converter = nbin.get_by_name(f"audioconv_{index}")
        if converter:
            sink_pad = converter.get_static_pad("sink")
            if not sink_pad.is_linked():
                if decoder_src_pad.link(sink_pad) != 0:
                    sys.stderr.write("Failed to link decoder src pad to source bin pad\n")
    elif gstname.find("video") != -1:
        features = caps.get_features(0)
        logger.info(f"features={features}")
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = nbin.get_static_pad("src-video")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.error("Failed to link decoder src pad to source bin ghost pad")
        else:
            logger.error(" Error: Decodebin did not pick nvidia decoder plugin.")


def decodebin_child_added(unused_child_proxy: Any, obj: Any, name: str, user_data: Any) -> None:
    """DecodeBin child-added callback handler."""
    logger.info(f"Decodebin child added: {name}")
    if name.find("decodebin") != -1:
        obj.connect("child-added", decodebin_child_added, user_data)


def source_setup(pipeline: Gst.Pipeline, source: Any, data: Any) -> None:
    if source.__class__.__name__ == "GstRTSPSrc":
        logger.info("Configuring GstRTSPSrc timeout")
        source.set_property("tcp-timeout", 200000000)
        source.set_property("teardown-timeout", 1000000000)
        source.set_property("timeout", 50000000)


def create_source_bin(index: int, uri: str, rate: int | None = 44100, loop: bool = False) -> Gst.Bin:
    """Create uridecodebin to abstract this bin's content from the rest of the
    pipeline."""
    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error(" Unable to create source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = make_elm_or_print_err(
        "nvurisrcbin" if loop else "uridecodebin", "uri-decode-bin", "uri decode bin"
    )

    if loop:
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)

    if rate:
        audio_convert = make_elm_or_print_err("audioconvert", f"audioconv_{index}", f"audioconv_{index}")
        audio_resample = make_elm_or_print_err("audioresample", f"audioresampler_{index}", f"audioresampler_{index}")
        cap_filter = make_elm_or_print_err(
            "capsfilter",
            f"cap_filter_audioresample_{index}",
            f"cap_filter_audioresample_{index}",
        )
        cap_filter.set_property(
            "caps",
            Gst.Caps.from_string(f"audio/x-raw, rate={rate}"),
            # format=S16LE, layout=interleaved, channels=1
        )

        for elem in [uri_decode_bin, audio_convert, audio_resample, cap_filter]:
            Gst.Bin.add(nbin, elem)

        audio_convert.link(audio_resample)
        audio_resample.link(cap_filter)

        src_pad = cap_filter.get_static_pad("src")
        bin_pad = nbin.add_pad(Gst.GhostPad.new("src-audio", src_pad))

        if not bin_pad:
            logger.error(" Failed to add ghost pad for audio in source bin")
            return None

    if USE_NEW_NVSTREAMMUX or FIX_FPS:
        nvvidconv = make_elm_or_print_err("nvvideoconvert", f"src_convertor_{index}", f"Nvvidconv_{index}")
        videorate = make_elm_or_print_err("videorate", f"src_rate_{index}", f"vidrate_{index}")
        caps = Gst.ElementFactory.make("capsfilter", f"src_capsfilter_{index}")
        caps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=25/1"),
        )

        for elem in [uri_decode_bin, nvvidconv, videorate, caps]:
            Gst.Bin.add(nbin, elem)

        uri_decode_bin.link(nvvidconv)
        nvvidconv.link(videorate)
        videorate.link(caps)
        src_pad = caps.get_static_pad("src")
        bin_pad = nbin.add_pad(Gst.GhostPad.new("src-video", src_pad))
    else:
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src-video", Gst.PadDirection.SRC))

    if not bin_pad:
        logger.error(" Failed to add ghost pad for video in source bin")
        return None

    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has been created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    return nbin


def create_gie(process_mode: str, name: str, config_file_path: str) -> Gst.Element:
    gie = make_elm_or_print_err("nvinfer", f"{process_mode}-inference-{name}", f"{process_mode}-gie-{name}")
    gie.set_property("config-file-path", config_file_path)
    return gie


def link_source_to_mux(bin: Gst.Bin, mux: Any, padname: str, typestr: str) -> None:
    sinkpad = mux.get_request_pad(padname)
    if not sinkpad:
        logger.error("Unable to create sink pad bin")
    srcpad = bin.get_static_pad(f"src-{typestr}")
    if not srcpad:
        logger.error("Unable to create src pad bin")
    srcpad.link(sinkpad)


def find_video_element(pipeline: Gst.Pipeline, name: str) -> Gst.Element:
    return pipeline.get_by_name(f"primary-inference-{name}")


def create_and_link_video_branch(
    pipeline: Gst.Pipeline,
    tee: Gst.Element,
    name: str,
    prep_config: str,
    vid_config: str,
    cb_func: Any,
    params: Any,
    sync: bool = False,
) -> Gst.Element:
    prep_queue = Gst.ElementFactory.make("queue", f"queue-preprocess-{name}")
    prep_vid = make_elm_or_print_err("nvdspreprocess", f"preprocess-{name}", f"preprocess-{name}")
    prep_vid.set_property("config-file", prep_config)

    vid_queue = Gst.ElementFactory.make("queue", f"queue-pgie-{name}")
    pgie_vid = make_elm_or_print_err("nvinfer", f"primary-inference-{name}", f"primary-gie-{name}")
    pgie_vid.set_property("config-file-path", vid_config)

    fakesink_vid = make_elm_or_print_err("fakesink", f"fakesink-{name}", f"fakesink-{name}")

    pipeline.add(prep_queue)
    pipeline.add(prep_vid)
    pipeline.add(vid_queue)
    pipeline.add(pgie_vid)
    pipeline.add(fakesink_vid)

    tee.link(prep_queue)
    prep_queue.link(prep_vid)
    prep_vid.link(vid_queue)
    vid_queue.link(pgie_vid)
    pgie_vid.link(fakesink_vid)

    if sync:
        fakesink_vid.set_property("sync", 1)

    pgie_vid_src_pad = pgie_vid.get_static_pad("src")
    if not pgie_vid_src_pad:
        logger.error(" Unable to get src pad of primary infer")
    pgie_vid_src_pad.add_probe(Gst.PadProbeType.BUFFER, cb_func, params)

    return pgie_vid


def log_message(timestamp: int, source_id: int, msg: str) -> None:
    logger.info(f"{timestamp}: source {source_id}: {msg}")


def get_attributes(obj_meta: Any) -> Any:
    attributes = OrderedDict()
    l_clf = obj_meta.classifier_meta_list
    while l_clf is not None:
        try:
            cls_meta = pyds.NvDsClassifierMeta.cast(l_clf.data)
            l_lbl = cls_meta.label_info_list
        except StopIteration:
            break

        while l_lbl is not None:
            try:
                label_info = pyds.NvDsLabelInfo.cast(l_lbl.data)
            except StopIteration:
                break

            attributes[label_info.result_label] = label_info.result_prob

            try:
                l_lbl = l_lbl.next
            except StopIteration:
                break

        try:
            l_clf = l_clf.next
        except StopIteration:
            break
    return attributes


def xy2wh(box: list[float]) -> NDArray[np.float32]:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.float32)


def plate2vehicle(plates: Any, vehicles: Any) -> NDArray[np.int64]:
    vboxes = np.array([xy2wh(vehicle["bbox"]) for vehicle in vehicles])
    pboxes = np.array([xy2wh(plate["bbox"]) for plate in plates])

    vboxes[:, :2] += vboxes[:, 2:] / 2
    pboxes[:, :2] += pboxes[:, 2:] / 2

    distances = cdist(vboxes[:, :2], pboxes[:, :2]) / (vboxes[:, 2:].max(axis=1, keepdims=True) * 0.5)
    return {pind: vind for vind, pind in np.asarray(list(zip(*linear_assignment(distances))))}


# import inspect


# def gstds_err_msg(msg: str) -> None:
#     frame = inspect.stack()[1][0]
#     info = inspect.getframeinfo(frame)
#     print(f"** ERROR: <{info.function}:{info.lineno}>: {msg}")


# def gstds_info_msg(msg: str) -> None:
#     frame = inspect.stack()[1][0]
#     info = inspect.getframeinfo(frame)
#     print(f"** INFO: <{info.function}:{info.lineno}>: {msg}")


# def gstds_warn_msg(msg: str) -> None:
#     frame = inspect.stack()[1][0]
#     info = inspect.getframeinfo(frame)
#     print(f"** WARN: <{info.function}:{info.lineno}>: {msg}")
