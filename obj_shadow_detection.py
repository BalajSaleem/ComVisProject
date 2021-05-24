from detectron2.config import get_cfg
from predictor import VisualizationDemo

def setup_cfg(config_file, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def get_obj_shadow_masks_predictor(image, config_file, confidence_threshold = 0.5):
    cfg = setup_cfg(config_file, confidence_threshold)
    predictor = VisualizationDemo(cfg)
    return predictor
