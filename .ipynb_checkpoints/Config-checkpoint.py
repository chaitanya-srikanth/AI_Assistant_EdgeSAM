
### MODEL IDS ####
# ORI_IMG_MODEL_ID = '66052a556f7b5bf90ccdd7a1'
AUG_IMG_MODEL_ID = '661e46428c1fb79d0566d5f0'


PHASE2_MODEL_ID = '661f763cfb55878931e5ebaf'
## MASTER COLLECTION ID ##
MASTER_COLLECTION_ID = '661e46778c1fb79d0566da43'

## MASTER COLLECTION NAME ##
MASTER_COLLECTION_NAME = 'Master'


### HYPERPARAMETER ###
HP = {
    "phase-1": { "project_type": "segmentation", "yolo_type": "s", "model_version": 8, "image_extention": "jpg", "epochs": 120, "batch": 16, "imgsz": 640, "gray": False, "single_cls": False, "aug_col": True, "workers": 8, "device": [ 0 ], "patience": 100, "integrity_confidence_threshold": 0.5 },
    "phase-2": { "project_type": "segmentation", "yolo_type": "s", "model_version": 8, "image_extention": "jpg", "epochs": 100, "batch": 8, "imgsz": 1080, "gray": False, "single_cls": False, "aug_col": True, "workers": 8, "device": [ 0 ], "patience": 100, "integrity_confidence_threshold": 0.7, "freeze" : 10 }
}
