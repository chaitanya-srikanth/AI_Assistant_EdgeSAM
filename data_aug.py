import random
import os
import numpy as np
import time
import uuid
import json
import cv2
import ast
import uuid 
import random
import time
import glob
import rlef_helper as rlh
import Config

class SimpleAugmentation:
    def __init__(self, image):

        self.image = image
    def brightnesscontrastImage(self):
        # control Contrast
        alpha = random.uniform(0.5, 1.5)
        # control brightness
        beta = random.randint(-30, 30)  
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        
    def sharpenImage(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.image = cv2.filter2D(self.image, -1, kernel)
    
    def blurImage(self):
        kernel = random.choice([3,5])
        self.image = cv2.GaussianBlur(self.image,(kernel,kernel),0)
        
    def hsvValues(self):
        # Convert the image from BGR to HSV color space 
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV) 

        # Adjust the hue, saturation, and value of the image 
        # Adjusts the hue by multiplying it
        img[:, :, 0] = img[:, :, 0]
        # Adjusts the saturation by multiplying it
        img[:, :, 1] = img[:, :, 1] * random.uniform(0.8, 1.5)
        # Adjusts the value by multiplying it
        img[:, :, 2] = img[:, :, 2] * random.uniform(0.7, 1)

        # Convert the image back to BGR color space 
        self.image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) 
    
    def modified_image(self):
        x = [True, False]
        
        command_list = [self.brightnesscontrastImage, self.sharpenImage, self.blurImage, self.hsvValues]
        random.shuffle(command_list)
        # print(command_list)
        for command in command_list:
            if random.choice(x):
                command()
        
        return self.image

    
    



class yoloRotate:
    def __init__(self, image, objects, width, height, angle):

        self.width = width
        self.height = height
        self.angle = angle
        
        # read image using cv2
        self.image = image
        self.objects = objects
        
        rotation_angle = self.angle*np.pi/180 # Converting angle(in degree) to radian
        
        #define a transformation matrix
        self.rot_matrix = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],
                                    [np.sin(rotation_angle),np.cos(rotation_angle)]])
        
#     def rotateYoloBB(self):
#         """
#         This function focuses on finding the new coordinates of the BB after Image Rotation.
#         """
#         new_ht, new_wd = self.rotate_image().shape[:2]         
        
#         # Read XML from data
#         # W, H, objects = read_xml_data(self.filename+'.xml') 
#         W = self.width
#         H = self.height
#         objects = self.objects
        
#         #Store Bounding Box coordinates from an image
#         new_bbox = []
        
#         # To store the differnt object classes, later used to generate the classes.txt file used for YOLO training.      
#         #[['fridge', 'fridge_handle'], [[46, 64, 322, 440], [47, 113, 66, 403]]]
#         # Convert XML data to YOLO format
#         for i,obj in enumerate(objects):
#             classes = obj[0]
            
#             print(obj)
#             xmin = obj[1][0]
#             ymin = obj[1][1]
#             xmax = obj[1][2] 
#             ymax = obj[1][3]    
            
#             # Calculating coordinates of corners of the Bounding Box
#             top_left = (xmin-W/2,-H/2+ymin)
#             top_right = (xmax-W/2,-H/2+ymin)
#             bottom_left = (xmin-W/2,-H/2+ymax)
#             bottom_right = (xmax-W/2,-H/2+ymax)
            
#             # Calculate new coordinates (after rotation)
#             new_top_left = []
#             new_bottom_right = [-1,-1]           

#             for j in (top_left,top_right,bottom_left,bottom_right):
#                 # Generate new corner coords by multiplying it with the transformation matrix (2,2)
#                 new_coords = np.matmul(self.rot_matrix, np.array((j[0],-j[1])))

#                 x_prime, y_prime = new_wd/2+new_coords[0],new_ht/2-new_coords[1]
                
#                 # Finding the new top-left coords, by finding the minimum of calculated x and y-values
#                 if len(new_top_left)>0:
#                     if new_top_left[0]>x_prime:
#                         new_top_left[0]=x_prime
#                     if new_top_left[1]>y_prime:
#                         new_top_left[1]=y_prime
#                 else:  # for first iteration, lists are empty, therefore directly append
#                     new_top_left.append(x_prime)
#                     new_top_left.append(y_prime)
                
#                 # Finding the new bottom-right coords, by finding the maximum of calculated x and y-values
#                 if new_bottom_right[0]<x_prime:
#                     new_bottom_right[0]=x_prime
#                 if new_bottom_right[1]<y_prime:
#                     new_bottom_right[1]=y_prime

#             # i(th) index of the object
#             new_bbox.append([new_top_left[0],new_top_left[1],new_bottom_right[0],new_bottom_right[1]])
            
#         return new_bbox
    def rotateYoloBB(self):
        """
        This function focuses on finding the new coordinates of the BB after Image Rotation.
        """
        new_ht, new_wd = self.rotate_image().shape[:2]         
        
        # Read XML from data
        # W, H, objects = read_xml_data(self.filename+'.xml') 
        W = self.width
        H = self.height
        objects = self.objects
        
        #Store Bounding Box coordinates from an image
        new_bbox = []
        
        # To store the differnt object classes, later used to generate the classes.txt file used for YOLO training.      
        
        # Convert XML data to YOLO format
        for i,obj in enumerate(objects):
            classes = obj[0]
            
            print(obj)
            xmin = obj[1][0]
            ymin = obj[1][1]
            xmax = obj[1][2] 
            ymax = obj[1][3]    
            
            # Calculating coordinates of corners of the Bounding Box
            top_left = (xmin-W/2,-H/2+ymin)
            top_right = (xmax-W/2,-H/2+ymin)
            bottom_left = (xmin-W/2,-H/2+ymax)
            bottom_right = (xmax-W/2,-H/2+ymax)
            
            # Calculate new coordinates (after rotation)
            new_top_left = []
            new_bottom_right = [-1,-1] 
            new_bbox = []

            for j in (top_left,top_right,bottom_left,bottom_right):
                # Generate new corner coords by multiplying it with the transformation matrix (2,2)
                new_coords = np.matmul(self.rot_matrix, np.array((j[0],-j[1])))

                x_prime, y_prime = new_wd/2+new_coords[0],new_ht/2-new_coords[1]
                new_bbox.append([x_prime, y_prime])
                
#                 # Finding the new top-left coords, by finding the minimum of calculated x and y-values
#                 if len(new_top_left)>0:
#                     if new_top_left[0]>x_prime:
#                         new_top_left[0]=x_prime
#                     if new_top_left[1]>y_prime:
#                         new_top_left[1]=y_prime
#                 else:  # for first iteration, lists are empty, therefore directly append
#                     new_top_left.append(x_prime)
#                     new_top_left.append(y_prime)
                
#                 # Finding the new bottom-right coords, by finding the maximum of calculated x and y-values
#                 if new_bottom_right[0]<x_prime:
#                     new_bottom_right[0]=x_prime
#                 if new_bottom_right[1]<y_prime:
#                     new_bottom_right[1]=y_prime
                

            # i(th) index of the object
            # new_bbox.append([new_top_left[0],new_top_left[1],new_bottom_right[0],new_bottom_right[1]])
            
        return new_bbox
    
    def rotate_image(self):
        """
        This function focuses on rotating the image to a particular angle. 
        """
        height, width = self.image.shape[:2]
        img_c = (width/2,height/2) # Image Center Coordinates
        
        rotation_matrix = cv2.getRotationMatrix2D(img_c,self.angle,1.) # Rotating Image along the actual center
        
        abs_cos = abs(rotation_matrix[0,0])  # Cos(angle)
        abs_sin = abs(rotation_matrix[0,1])  # sin(angle)
        
        # New Width and Height of Image after rotation
        bound_w = int(height * abs_sin+width*abs_cos)
        bound_h = int(height * abs_cos+width*abs_sin)
        
        # subtract the old image center and add the new center coordinates
        rotation_matrix[0,2]+=bound_w/2-img_c[0]
        rotation_matrix[1,2]+=bound_h/2-img_c[1]
        
        # rotating image with transformed matrix and new center coordinates
        rotated_matrix = cv2.warpAffine(self.image, rotation_matrix,(bound_w, bound_h))
        
        return rotated_matrix


ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def find_left_upper_right_down(points):
    if not points:
        return None, None  # Return None if the list is empty

    left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
    right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]

    return left_upper, right_down

def rlef_annotations_to_bbox(raw_annotations, height, width):
    try:
        dictionary = json.loads(raw_annotations)
    except:
        print("Error reading annotations")
    # print(dictionary[0]['selectedOptions'])
    # print('###########')
    final_annotations = []
    # final_class = []
    for idx, entry in enumerate(dictionary):
        vertex_list = []
        clss_value = entry['selectedOptions'][1]['value']
        vertices = entry['vertices']
        for point in vertices:
            vertex_list.append([point['x'], point['y']])
        left, right = find_left_upper_right_down(vertex_list)
        h,w = height, width
        # final_annotations.append(left[0]/w, left[1]/h, right[0], right[1])

        final_annotations.append([clss_value, [left[0], left[1], right[0], right[1]]])
        
    return final_annotations









def augment_images(image_path, objects, generated_count, model_id, label):
    
    output_dir = f"output/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    image = cv2.imread(image_path, 1)
    height, width = image.shape[:2]

    
    filename = os.path.basename(image_path)
    file = filename.split(".")
    image_name = file[0]
    image_ext = "."+file[1]
    aug_resource_ids = []
    
    
    # collection_id = rlh.create_collection(model_id, label)
    
    # print(f'{collection_id} : {label}')       
    for i in range(generated_count):
        angle = random.randint(0, 360)
        im = yoloRotate(image, objects, width, height, angle)
        bbox = im.rotateYoloBB()  # new BBox values, after rotation
        rotated_image= im.rotate_image()  # rotated_image
        print(bbox)
        aug = SimpleAugmentation(rotated_image)
        modified_img = aug.modified_image()
        
        out_name = str(uuid.uuid1())

        out_path = os.path.join(output_dir, out_name+image_ext)

        cv2.imwrite(out_path,modified_img) 
        rlef_format = rlh.pointPath_To_WeirdAutoaiAnnotationFormat(bbox, label)
        resource_id = rlh.send_to_rlef(img_path = out_path, model_id ="663caeaddc15f50ad142073d" , tag= "aug_data",label = "augmented_image", status = 'approved', annotation = rlef_format, confidence_score=100, prediction='predicted')
        
        os.remove(out_path)
        
        aug_resource_ids.append(resource_id)
    collection_id = "663dcba20778eb81b856d7ee"
    rlh.add_resource_to_collection("663dcba20778eb81b856d7ee", aug_resource_ids)
    return aug_resource_ids , collection_id
