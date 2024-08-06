import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

CHECKPOINT_PATH = 'weights/sam_vit_h_4b8939.pth'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)




mask_predictor = SamPredictor(sam)


def image_mask(IMAGE_PATH, box):
    '''
        box : np.array([xmin,ymin,xmax,ymax])
        
    '''
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    # box = np.array([70, 247, 626, 926])
    box = np.array(box)
    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=False
    )
    return masks[0]