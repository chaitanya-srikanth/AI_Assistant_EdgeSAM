import sys
import numpy as np
sys.path.append("..")
from EdgeSAM.edge_sam import sam_model_registry, SamPredictor

sam_checkpoint = "EdgeSAM/weights/edge_sam.pth"
model_type = "edge_sam"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def save_masked_image(mask, image_rgb):
    inv_final_mask = ~mask
    test_image = image_rgb.copy()
    test_image_arr = np.array(test_image)
    black_image = np.ones(test_image_arr.shape)*0
    masked_out_image = np.copy(test_image_arr)
    masked_out_image[inv_final_mask] = [0, 0, 0]
    return masked_out_image


def save_bg_removed_image(input_box, image_rgb):
    predictor.set_image(np.array(image_rgb))
    
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        num_multimask_outputs=1,
    )
    
    masked_image = save_masked_image(masks[0], image_rgb)
    return masked_image