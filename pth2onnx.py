from __future__ import annotations

import argparse
import typing
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import get_latest_opset, smart_inference_mode


def _forward(self: typing.Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenates and returns predicted bounding boxes and class
    probabilities."""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x
    elif self.dynamic or self.shape != shape:
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.export and self.format in (
        "saved_model",
        "pb",
        "tflite",
        "edgetpu",
        "tfjs",
    ):  # avoid TF FlexSplitV ops
        box = x_cat[:, : self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4 :]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides

    # BCN -> BNC
    return (
        dbox.permute(0, 2, 1).contiguous(),
        cls.sigmoid().permute(0, 2, 1).contiguous(),
    )


Detect.forward = _forward


@typing.no_type_check
@smart_inference_mode()
def export_onnx(args: typing.Any) -> None:
    yolo = YOLO(args.weights)
    model = yolo.model

    im = torch.zeros(1, 3, args.imgsz[0], args.imgsz[1])

    model.eval()
    model = model.fuse()

    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.export = True
            m.format = "onnx"

    for _ in range(2):
        _ = model(im)  # dry runs

    file = Path(args.weights)
    f = str(file.with_suffix(".onnx"))

    # Warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

    torch.onnx.export(
        model,
        im,
        f,
        verbose=False,
        opset_version=args.opset or get_latest_opset(),
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["input"],
        output_names=["boxes", "confs"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "confs": {0: "batch_size"},
        },
    )
    with open("labels.txt", "w") as f:
        for _, value in model.names.items():
            f.write(value + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="weights path")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    # parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    # parser.add_argument(
    #     "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    # )
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes")

    args = parser.parse_args()

    export_onnx(args)
