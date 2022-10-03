import copy
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils.misc import deprecated_api_warning
from PIL import Image

from mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose, ToTensor
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_pose_model(config, checkpoint=None, device="cuda:0"):
    """Initialize a pose model from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location="cpu")
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _pipeline_gpu_speedup(pipeline, device):
    """Load images to GPU and speed up the data transforms in pipelines.
    Args:
        pipeline: A instance of `Compose`.
        device: A string or torch.device.
    Examples:
        _pipeline_gpu_speedup(test_pipeline, 'cuda:0')
    """

    for t in pipeline.transforms:
        if isinstance(t, ToTensor):
            t.device = device


def _inference_single_pose_model(
    model,
    imgs_or_paths,
    bboxes,
    dataset="TopDownCocoDataset",
    dataset_info=None,
    return_heatmap=False,
    use_multi_frames=False,
    frame_weight_ind=False,
):
    """Inference human bounding boxes.
    Note:
        - num_frames: F
        - num_bboxes: N
        - num_keypoints: K
    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (list(str) | list(np.ndarray)): Image filename(s) or
            loaded image(s)
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name. Deprecated.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool): Flag to return heatmap, default: False
        use_multi_frames (bool): Flag to use multi frames for inference
    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device
    if device.type == "cpu":
        device = -1

    if not frame_weight_ind:
        # _test_pipeline[-1]['meta_keys'].remove('frame_weight')
        cfg.data.test.data_cfg["frame_weight_test"] = (1,)
    else:
        cfg.data.test.data_cfg["frame_weight_test"] = (0.3, 0.1, 0.1, 0.25, 0.25)

    if use_multi_frames:
        assert "frame_weight_test" in cfg.data.test.data_cfg
        # use multi frames for inference
        # the number of input frames must equal to frame weight in the config
        assert len(imgs_or_paths) == len(cfg.data.test.data_cfg.frame_weight_test)

    # build the data pipeline
    _test_pipeline = copy.deepcopy(cfg.test_pipeline)

    has_bbox_xywh2cs = False
    for transform in _test_pipeline:
        if transform["type"] == "TopDownGetBboxCenterScale":
            has_bbox_xywh2cs = True
            break
    if not has_bbox_xywh2cs:
        _test_pipeline.insert(0, dict(type="TopDownGetBboxCenterScale", padding=1.25))
    test_pipeline = Compose(_test_pipeline)
    _pipeline_gpu_speedup(test_pipeline, next(model.parameters()).device)

    assert len(bboxes[0]) in [4, 5]

    if dataset_info is not None:
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs

    batch_data = []
    for bbox in bboxes:
        # prepare data
        data = {
            "bbox": bbox,
            "bbox_score": bbox[4] if len(bbox) == 5 else 1,
            "bbox_id": 0,
            "dataset": dataset_name,
            "joints_3d": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            "joints_3d_visible": np.zeros(
                (cfg.data_cfg.num_joints, 3), dtype=np.float32
            ),
            "rotation": 0,
            "ann_info": {
                "image_size": np.array(cfg.data_cfg["image_size"]),
                "num_joints": cfg.data_cfg["num_joints"],
                "flip_pairs": flip_pairs,
            },
            "frame_weight": cfg.data.test.data_cfg.frame_weight_test,
        }

        if isinstance(imgs_or_paths[0], np.ndarray):
            if not frame_weight_ind:
                data["img"] = [imgs_or_paths]
            else:
                data["img"] = imgs_or_paths
        else:
            data["image_file"] = imgs_or_paths

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    batch_data = scatter(batch_data, [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data["img"],
            img_metas=batch_data["img_metas"],
            return_loss=False,
            return_heatmap=return_heatmap,
        )

    return result["preds"], result["output_heatmap"]


@deprecated_api_warning(name_dict=dict(img_or_path="imgs_or_paths"))
def inference_top_down_pose_model(
    model,
    imgs_or_paths,
    person_results=None,
    bbox_thr=None,
    format="xywh",
    dataset="TopDownCocoDataset",
    dataset_info=None,
    return_heatmap=False,
    outputs=None,
    frame_weight_ind=False,
):
    """Inference a single image with a list of person bounding boxes. Support
    single-frame and multi-frame inference setting.
    Note:
        - num_frames: F
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W
    Args:
        model (nn.Module): The loaded pose model.
        imgs_or_paths (str | np.ndarray | list(str) | list(np.ndarray)):
            Image filename(s) or loaded image(s).
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:
            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.
    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # decide whether to use multi frames for inference
    if isinstance(imgs_or_paths, (list, tuple)):
        use_multi_frames = True
    else:
        assert isinstance(imgs_or_paths, (str, np.ndarray))
        use_multi_frames = False
    # get dataset info
    if dataset_info is None and hasattr(model, "cfg") and "dataset_info" in model.cfg:
        dataset_info = DatasetInfo(model.cfg.dataset_info)
    if dataset_info is None:
        warnings.warn(
            "dataset is deprecated."
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663"
            " for details.",
            DeprecationWarning,
        )

    # only two kinds of bbox format is supported.
    assert format in ["xyxy", "xywh"]

    pose_results = []
    returned_outputs = []

    if person_results is None:
        # create dummy person results
        sample = imgs_or_paths[0] if use_multi_frames else imgs_or_paths
        if isinstance(sample, str):
            width, height = Image.open(sample).size
        else:
            height, width = sample.shape[:2]
        person_results = [{"bbox": np.array([0, 0, width, height])}]

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box["bbox"] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == "xyxy":
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = bbox_xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_pose_model(
            model,
            imgs_or_paths,
            bboxes_xywh,
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            use_multi_frames=use_multi_frames,
            frame_weight_ind=frame_weight_ind,
        )

        if return_heatmap:
            h.layer_outputs["heatmap"] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy)
    )
    for pose, person_result, bbox_xyxy in zip(poses, person_results, bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result["keypoints"] = pose
        pose_result["bbox"] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results, returned_outputs
