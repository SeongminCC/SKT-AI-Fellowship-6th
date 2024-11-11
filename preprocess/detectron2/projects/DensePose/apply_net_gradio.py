#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List

import cv2
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import DensePoseOutputsExtractor, DensePoseResultExtractor
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from PIL import Image


class DensePose4Gradio:
    def __init__(self, cfg, model) -> None:
        cfg = self.setup_config(cfg, model, [])
        self.predictor = DefaultPredictor(cfg)

    def setup_config(
        self, config_fpath: str, model_fpath: str, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    def execute(self, image: Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            return self.execute_on_outputs(img, outputs)

    def execute_on_outputs(self, image: np.ndarray, outputs: Instances):
        result = {}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

                H, W, _ = image.shape

                i = result['pred_densepose'][0].labels.cpu().numpy()
                i_scale = (i.astype(np.float32) * 255 / 24).astype(np.uint8)
                i_color = cv2.applyColorMap(i_scale, cv2.COLORMAP_PARULA)
                i_color = cv2.cvtColor(i_color, cv2.COLOR_RGB2BGR)
                i_color[i == 0] = [0, 0, 0]

                box = result["pred_boxes_XYXY"][0]
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                x, y, w, h = [int(v) for v in box]

                bg = np.zeros((H, W, 3))
                bg[y:y + h, x:x + w, :] = i_color

                bg_img = Image.fromarray(np.uint8(bg), "RGB")

                return bg_img
