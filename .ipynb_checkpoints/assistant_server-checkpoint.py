import cv2
import json
import random
import math
import operator
from functools import reduce 
from ultralytics import YOLO
import requests
from flask import Flask, request, jsonify
import traceback
from datetime import date
import os
from pprint import pprint
import numpy as np
import time
import uuid
import json
from pprint import pprint
from PIL import Image
from dotenv import load_dotenv
import cv2
import requests
from edge_sam_helper import save_bg_removed_image
from data_aug import augment_images
import Config
import rlef_helper as rlh
from concurrent.futures import ThreadPoolExecutor 


executor = ThreadPoolExecutor(max_workers = 2)
# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Variables
ALLOWED_EXTENSIONS = ['png', 'jpg']

# Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






def find_left_upper_right_down(points):
    if not points:
        return None, None  # Return None if the list is empty

    left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
    right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]

    return [left_upper[0], left_upper[1], right_down[0], right_down[1]]
def calculate_area(shape):
    area = 0
    n = len(shape)
    for i in range(n):
        x1, y1 = shape[i]
        x2, y2 = shape[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2



def maskProcessor(masks, tolerance = 2):
    threshold_value = 254
    binary_mask  = np.uint8(masks) * 255 
    #mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary_mask, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Simplify the contour using the Ramer-Douglas-Peucker algorithm
    
    # adjust this value to control the level of simplification
    simplified_contours = []
    for contour in contours:
        simplified_contour = cv2.approxPolyDP(contour, tolerance, True)
        if simplified_contour.shape[0] > 1:
            simplified_contour = simplified_contour.squeeze()
            simplified_contours.append(simplified_contour.tolist())
            
    points = [shape for shape in simplified_contours if len(shape) >= 3]
    

    
    points = sorted(points, key=calculate_area, reverse=True)
    largest_area = max(points, key=calculate_area)
    largest_area_value = calculate_area(largest_area)

    # Update the points list by removing shapes with area less than 1% compared to the largest shape
    points = [shape for shape in points if (calculate_area(shape) / largest_area_value) * 100 >= 0.75]
    return points



def json_creater(inputs, closed):
    data = []
    count = 1
    highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                             'rgba(227,0,255,1)']
    for index, input in enumerate(inputs):
        color = random.sample(highContrastingColors, 1)[0]
        json_id = count
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input]}]
        points = eval(input)
        if len(points) > 0:
            center = tuple(
                map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
            sorted_coords = sorted(points, key=lambda coord: (-135 - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        else:
            sorted_coords = []
        vertices = []
        is_first = True
        for vertex in sorted_coords:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = count
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[1]
            vertex_json['y'] = vertex[0]
            vertices.append(vertex_json)
            count += 1
        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    return json.dumps(data)


# def assistant_process(image_path):
   
@app.route('/')
def home():
    return "AI-Assistant server is available"




def send_image_annotation(model_id, tag, label, annotation, filename):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': 'backlog',
        # 'csv': f"{csv}",
        'label': label,
        'tag': tag,
        'model_type': 'imageAnnotation',
        'prediction': label,
        'confidence_score': '100',
        'imageAnnotations': str(annotation)
    }
    files = [('resource', (filename, open((filename), 'rb'), 'image/png'))]
    headers = {}
    requests.request("POST", url, headers=headers, data=payload, files=files) 

    
    
def parse_rlef_annotations(image_annotations):
    image_annotations = json.loads(image_annotations)
    all_annotations = []
    for annotation in image_annotations:
        bbox = []
        for vertex in annotation['vertices']:
            # print(vertex)
            x = int(vertex['x'])
            y = int(vertex['y'])
            bbox.append([x,y])
        all_annotations.append(bbox)
    return all_annotations
    
            
@app.route('/get_collection_ids', methods = ['GET'])
def get_collection_ids():
    try:
        results = rlh.get_all_collections(Config.AUG_IMG_MODEL_ID)
        return {'results': results}, 200
    except Exception as e:
        print(f'{e} at line 76, in get_collection_ids')
        return {'status': "Failed"}, 400     
            
    
@app.route('/aihelp', methods=["POST"])
def make_prediction():
    
    
    file = request.files['resource']
    
    if 'imageAnnotations' not in request.form.keys():
        return {'status': 'Image does not contain annotations'}, 400
    
    
    
    print('################## PROCESSING STARTED #####################')
    try:
        image_annotations = request.form.get('imageAnnotations')
    
        bboxes = parse_rlef_annotations(image_annotations)
        
        if file and allowed_file(file.filename):
            file_id = uuid.uuid1().hex
            filename = file_id + '.' + file.filename.split('.')[-1]
            file.save(filename)
        else:
            return 'Invalid file type', 500  

        image = cv2.imread(filename)
        image_rgb = image.copy()
        
        '''
        Mostly, images for phase-1 data augmentation will contain only one annotation per image. So, xyxy[0] is used.
        If Images have more than one box, code should be changed accordingly. 
        Edge SAM helper should be give a list of boxes, it will generate the mask accordingly......
        All the bounding boxes should be iterated through
        '''
    
        xyxy = []
        for box in bboxes:
            points = find_left_upper_right_down(box)
            xyxy.append(points)
        
        ## THIS PART SHOULD BE ITERATED ##
        
        input_box = np.array(xyxy[0])
        print(f'######### INPUT BOX : {input_box} #################')
        masked_image = save_bg_removed_image(input_box, image_rgb)
        
        if os.path.exists('masked_images') is not True:
            os.mkdir('masked_images')
        image_path = f'masked_images/{filename}'
        cv2.imwrite(image_path, masked_image)
        
        rlh.send_to_rlef(image_path, Config.PHASE2_MODEL_ID , 'phase-2','bg-removed', 'approved', image_annotations)

        
        objects = [['object', xyxy[0]]]
        generated_count = 10 
        label = 'object'
        
        ## STARTING DATA AUGMENTATION PROCESS ##
        # executor.submit(augment_images, image_path, objects, generated_count, Config.AUG_IMG_MODEL_ID, label)
        augment_images(image_path, objects, generated_count, Config.AUG_IMG_MODEL_ID, label)
        
        
        response_packet = {
             'csv' : "Background removal done",
             'metadata': 'Background Removal done'
        }
        return jsonify(response_packet), 200

    except Exception as e:
        print('Detailed Error')
        print(traceback.format_exc())
        return "Error : %s" % e, 500
    finally:
        os.remove(filename)
        
        
if __name__ == "__main__":
    print('Ping at ->')
    os.system('curl ipinfo.io/ip ; echo')
    app.run(host='0.0.0.0', port=8505, debug=True)