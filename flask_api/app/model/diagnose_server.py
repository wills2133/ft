
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
from sklearn.externals import joblib
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import dlib
import os
import shutil
import matplotlib.pyplot as plt
from collections import Counter
import copy
import urllib
import leancloud
import pytz
import datetime
import multiprocessing
import time
import json


# In[4]:


debug = False
notebook = 0
leancloudid = 'm6lOVETdPhWIYo697GSYejad-gzGzoHsz'
leancloudkey = 'NUEx4YnjBk9nrQmWv7zDFtYm'
if notebook:
    import tongue_diagnose as tgd
    dlib_path = './models/shape_predictor_68_face_landmarks.dat'
    model_dir = './models'
    output_img_dir = './result/'
    input_face_path = output_img_dir + '1.jpg'
    input_tongue_path = output_img_dir + '2.jpg'
    tongue_pic_path = './result/tongue.jpg'
    tongue_mask_path = './result/tongue_mask.png'
    json_dir = './'
else:
    from . import tongue_diagnose as tgd
    dlib_path = './app/model/models/shape_predictor_68_face_landmarks.dat'
    model_dir = './app/model/models'
    output_img_dir = './app/resource/'
    input_face_path = output_img_dir + '1.jpg'
    input_tongue_path = output_img_dir + '2.jpg'
    tongue_pic_path = output_img_dir + 'tongue.jpg'
    tongue_mask_path = output_img_dir + 'tongue_mask.png'
    json_dir = './app/model/'


# In[5]:


def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(30,30), ratio=1, conv_color = False):
    
    if conv_color and len(list_of_images[0].shape)>2:
        list_of_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in list_of_images]
    
    num_img = len(list_of_images)
    row = math.ceil( num_img / no_of_columns )
    
    if figsize[1] == 0:
        fig = plt.figure(figsize=(figsize[0], figsize[0]*row/no_of_columns/ratio))
    else:
        fig = plt.figure(figsize=figsize)
        
    for i, img in enumerate(list_of_images):
        if list_of_titles:
            fig.add_subplot( row, no_of_columns , i+1, title = list_of_titles[i])
        else:
            fig.add_subplot( row, no_of_columns , i+1)
        plt.imshow(list_of_images[i])
        plt.axis('off')
    plt.show()


# In[6]:


def get_lankmark(path, landmark_predictor):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(path)
    img_shape = img.shape
    faces = detector(img, 1) #detect face
    landmark = []
    rect = []
    if (len(faces) > 0):
        for k, d in enumerate(faces):
            rect.append( [( max(d.left(), 10), max(d.top(), 10) ), 
                          ( min(d.right(), img_shape[1]-10),  min(d.bottom(), img_shape[0]-10) )] )
            
            shape = landmark_predictor(img, d)
            feas = []  #lankmark points
            for i in range(68):
                num = str(shape.part(i))[1:-1].split(",")
                feas.append( (int(num[0]), int(num[1])) )
            feas.append( rect[k][0] ) 
            feas.append( rect[k][1] ) 
            landmark.append(feas)
    else:
        raise Exception('No face is detected in the photo!')
    
    return img, rect, landmark, img_shape


# In[7]:


def get_organ_boxes(landmark, img_h, img_w):
    #define boxes coordinate
    fix_ratio = 1.1
    #cheek_l
    cheek_l_x_left = min(landmark[0][1][0], landmark[0][2][0], landmark[0][3][0]) #x_left_top
    cheek_l_x_left = max(cheek_l_x_left, 15)
    cheek_l_y_top = max(landmark[0][36][1], landmark[0][39][1], landmark[0][40][1], landmark[0][41][1]) #y_left_top
    cheek_l_x_right = (landmark[0][31][0] + landmark[0][31][0] - landmark[0][32][0]) #x_right_bottom
    cheek_l_y_bottom = cheek_l_y_top + round( (cheek_l_x_right - cheek_l_x_left) * fix_ratio ) #y_right_bottom landmark[0][49][1]
#     print ('coodrinate:{} {} {} {}'.format(
#         cheek_l_x_left, 
#         cheek_l_y_top, 
#         cheek_l_x_right, 
#         cheek_l_y_bottom,))
    #cheek_r
    cheek_r_x_left = (landmark[0][35][0]+landmark[0][35][0]-landmark[0][34][0]) #x_left_top
    cheek_r_y_top = max(landmark[0][46][1], landmark[0][47][1], landmark[0][45][1], landmark[0][42][1]) #y_left_top
    cheek_r_x_right = min(landmark[0][15][0], img_w-30) #x_right_bottom
    cheek_r_y_bottom = cheek_r_y_top + round( (cheek_r_x_right - cheek_r_x_left) * fix_ratio ) #y_right_bottom landmark[0][63][1]
    #jaw
    jaw_y_top = landmark[0][57][1] #y_top
    jaw_y_bottom = min(landmark[0][8][1], img_h)#y_bottom
    jaw_w_middle = round( (landmark[0][48][0] + landmark[0][54][0]) // 2 )
    jaw_h_offset = round( (jaw_y_bottom - jaw_y_top) * fix_ratio // 2 )
    jaw_x_left = jaw_w_middle - jaw_h_offset #x_left landmark[0][5][0]
    jaw_x_right = jaw_w_middle + jaw_h_offset #x_right landmark[0][11][0]
    #forehead
    forehead_y_top = landmark[0][19][1] - (landmark[0][23][0] - landmark[0][20][0])  #y_left_top
    forehead_y_top = max (forehead_y_top, 10)
    forehead_y_bottom = min(landmark[0][17][1], landmark[0][18][1], landmark[0][19][1], landmark[0][20][1]) #y_right_bottom
    forehead_w_middle = round( (landmark[0][24][0] + landmark[0][19][0]) // 2 )
    forehead_h_offset = round( (forehead_y_bottom - forehead_y_top) * fix_ratio // 2 )
    forehead_x_left = forehead_w_middle - forehead_h_offset #x_left_top landmark[0][19][0]
    forehead_x_right = forehead_w_middle + forehead_h_offset #x_right_bottom
    #nose
    nose_x_left = (landmark[0][31][0] + landmark[0][31][0] - landmark[0][32][0]) #x_left_top
    nose_x_right = (landmark[0][35][0] + landmark[0][35][0] - landmark[0][34][0]) #x_right_bottom
    nose_y_bottom = max(landmark[0][31][1], landmark[0][32][1], landmark[0][33][1], landmark[0][34][1], landmark[0][35][1]) #y_right_bottom
    nose_y_top = nose_y_bottom - round( (nose_x_right - nose_x_left) * fix_ratio ) #y_left_top landmark[0][27][1]
    #tongue
    tongue_x_left = landmark[0][48][0]-10 #x_left_top
    tongue_y_top = min(landmark[0][48][1], landmark[0][54][1])  #y_left_top
    tongue_x_right = landmark[0][54][0]+10 #x_right_bottom
    
    tongue_y_right_bottom = max(landmark[0][7][1], landmark[0][9][1], landmark[0][8][1])
    tongue_y_right_bottom = tongue_y_right_bottom + 20 if img_h - tongue_y_right_bottom > 20 else img_h
#     print (f'tongue_y_right_bottom {tongue_y_right_bottom}')
    tongue_y_bottom = tongue_y_right_bottom #y_right_bottom
    
    return {
    'face':(
        landmark[0][-2][0], #x_left_top
        landmark[0][-2][1],  #y_left_top
        landmark[0][-1][0], #x_right_bottom
        landmark[0][-1][1], #y_right_bottom
    ),
    'forehead':(
        forehead_x_left, forehead_y_top, forehead_x_right, forehead_y_bottom,
    ),
    'cheek_l':(
        cheek_l_x_left, cheek_l_y_top, cheek_l_x_right, cheek_l_y_bottom,
    ),
    'cheek_r':(
        cheek_r_x_left, cheek_r_y_top, cheek_r_x_right, cheek_r_y_bottom,
    ),
    'nose':(
        nose_x_left, nose_y_top, nose_x_right, nose_y_bottom,
    ),
    'lip':(
        landmark[0][48][0], #x_left_top
        min(landmark[0][50][1], landmark[0][52][1]),  #y_left_top
        landmark[0][54][0], #x_right_bottom
        max(landmark[0][56][1], landmark[0][57][1], landmark[0][58][1]), #y_right_bottom
    ),
    'jaw': (
        jaw_x_left, jaw_y_top, jaw_x_right, jaw_y_bottom,
    ),
#     'neck':(
#         landmark[0][5][0], #x_left_top
#         landmark[0][8][1], #y_left_top
#         landmark[0][11][0], #x_right_bottom
#         landmark[0][8][1] + landmark[0][8][1] - landmark[0][57][1], #y_right_bottom
#     ),
    'eyes':(
        landmark[0][36][0]-20, #x_left_top
        min(landmark[0][37][1], landmark[0][38][1], landmark[0][43][1], landmark[0][44][1]) - 10, #y_left_top
        landmark[0][45][0]+20, #x_right_bottom
        max(landmark[0][40][1], landmark[0][41][1], landmark[0][46][1], landmark[0][47][1]) + 10, #y_right_bottom
    ),
    'tongue':(
        tongue_x_left, tongue_y_top, tongue_x_right, tongue_y_bottom,
    ),
    }


# In[8]:


def mask_leye(img, landmark):
    # extract lip shape
    # get countour from landmark
    countour = [np.array(landmark[0][36:41])] 
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(img.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, countour, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    # apply the mask
    img_masked = cv2.bitwise_and(img, mask)
    return img_masked


# In[9]:


def detect_face(obj_img_path, organ_dirs, organ_resize, model_dlib):
    #extract landmark features
    img, rect, landmark, img_shape  = get_lankmark(obj_img_path, model_dlib)
#     img, rect, landmark, img_shape  = get_lankmark('./test/2345.jpg')
#     img, rect, landmark, img_shape = get_lankmark('./test/f-026-01.jpg')
#     img, rect, landmark = get_lankmark(sample_dir + face_img)
    organ_boxes = get_organ_boxes(landmark, img_shape[0], img_shape[1])
    # add facelandmark
#     organ_boxes['face'] = (rect[0][0][0], rect[0][0][1], rect[0][1][0], rect[0][1][1])
    
    img_organs={}
    # cut different parts of face
    for organ in organ_dirs:
#         print (organ)
        if debug:
            img_orig = copy.copy(img)
        organ_crop = img[ organ_boxes[organ][1]:organ_boxes[organ][3],
                                                 organ_boxes[organ][0]:organ_boxes[organ][2] ]
        
#         cv2.imwrite(organ_dirs[organ] + face_img, organ_crop)
        # save organ crops
#         print (organ)
#         print (organ_crop.shape)
#         print (organ+'.jpg')
        cv2.imwrite(output_img_dir+organ+'.jpg', organ_crop)
        
        if organ == 'eyes' or organ == 'lip':
            img_organs[organ] = organ_crop
        else:
            img_organs[organ] = cv2.resize(organ_crop, organ_resize)

    if debug:
        # draw face rect and landmark features
        for i, feat_point in enumerate(landmark[0]):
            cv2.circle(img_orig, feat_point, 2, (0, 0, 0))
            cv2.putText(img_orig, str(i), feat_point, 3, 0.7, (0, 0, 255))
        for organ in organ_dirs:
            cv2.rectangle(img_orig, (organ_boxes[organ][0], organ_boxes[organ][1]), (organ_boxes[organ][2],organ_boxes[organ][3]),
                          (255, 255, 255), 2)
        cv2.rectangle(img_orig, rect[0][0], rect[0][1], (0, 0, 255), 2)
#         print(f'face {rect[0][0]} {rect[0][1]}')
        print (f'imgh {img_shape[0]}, imgw {img_shape[1]}')
#         img_eye = mask_leye(img, landmark)
        
        imgs=[]
        # resize image
        img_s = cv2.resize(img_orig, (0,0), fx=0.3, fy=0.3)
        imgs.append(img_s)
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in imgs]
        grid_display(images, [], 1, (30,30), 1, False)
    
    return img_organs, organ_boxes, img_shape, landmark


# ### generate face landmark to view

# In[10]:


def gener_svg():
    landmark_range = {
        'face_svg' : range(0,17),
        'eye_l_svg1':range(36,40),
        'eye_l_svg2':[39,40,41,36],
        'eye_r_svg1' : range(42,46),
        'eye_r_svg2' : [45,46,47,42],
        'lig_svg1' : range(48,55),
        'lig_svg2' : [54,55,56,57,58,59,48],
        'lig_svg3' : [48,60,61,62,63,54],
        'nose_svg1' : range(27,31),
        'nose_svg2' : range(31,36),
    }

    contour_path = [] 

    for key in landmark_range:
        svg_curve, svg_points = generate_svg(landmark_range[key])
        contour_path.append(svg_curve)
    if debug:
        print (contour_path)


# In[11]:


def generate_svg(landmark_range):
    # print (landmark[0][1:16])
    divisor = 2
    start = landmark_range[0]
    svg_curve = 'M {:.0f},{:.0f} '.format(landmark[0][start][0] / divisor, landmark[0][start][1] / divisor)
    svg_points = [{'cx': landmark[0][start][0] / divisor, 'cy': landmark[0][start][1] / divisor}]
    for i, _ in enumerate(landmark_range[:-1]):
        former = landmark_range[i]
        later = landmark_range[i+1]
        offset_x = -5 if i < 7 else 5
        offset_y = 5
        start_x = landmark[0][former][0] / divisor
        start_y = landmark[0][former][1] / divisor
        end_x = landmark[0][later][0] / divisor
        end_y = landmark[0][later][1] / divisor
        curve_x = (start_x + end_x)/2
        curve_y = (start_y + end_y)/2
        svg_points.append({'cx':end_x, 'cy':end_y})
        svg_curve = svg_curve + 'Q {:.0f},{:.0f}, {:.0f},{:.0f} '.format(
            curve_x, curve_y, end_x, end_y)
    return svg_curve, svg_points


# ### generate  organ position data to view

# In[12]:


# print(organ_boxes)
# print(img_shape)
def get_organ_position(organ_boxes, img_shape, move_direction):
    img_w = img_shape[1]
    img_h = img_shape[0]
    organ_position = {}
    for key in organ_boxes:
        w_perc = (organ_boxes[key][2] - organ_boxes[key][0]) / img_w * 100
        h_perc = (organ_boxes[key][3] - organ_boxes[key][1]) / img_h * 100
        x_perc = organ_boxes[key][0] / img_w * 100
        y_perc = (img_h - organ_boxes[key][3]) / img_h * 100
        x_perc1 = move_direction * (150 + x_perc) * 100 / w_perc # move horizontally on the ratio of self size x
        y_perc1 = (y_perc - 50) * 100/ h_perc
#         end_x = (1.5 - organ_boxes[key][0]/img_w) * 10000/ w_perc
#         end_y = (organ_boxes[key][3]/img_h - 0.5) * 10000/ h_perc
        end_x = 0
        end_y = 0
        if key == 'face': # to draw face rectangule
            x_perc1 = y_perc1 = end_x = end_y = 0
#         key: [width%, height%, start_left%, start_bottom%, middle_left%, middle_bottom&%, end_left%, end_bottom%]
        organ_position[key] = ["{:.0f}%".format(w_perc), "{:.0f}%".format(h_perc), 
                               "{:.0f}%".format(x_perc), "{:.0f}%".format(y_perc), 
                               "{:.0f}%".format(x_perc1), "{:.0f}%".format(y_perc1),
                               "{:.0f}%".format(end_x), "{:.0f}%".format(end_y)]
    return organ_position
        
# organ_position = get_organ_position()
# print (organ_position)


# ### generate hist data

# In[13]:


def get_hist(organ_img, dim):
    img = organ_img
    img = cv2.resize(img, (120,120))
    blue, green, red = img[:,:,0], img[:,:,1], img[:,:,2]
    # normailize
#     img[:,:,0] = blue/256
#     img[:,:,1] = green/256
#     img[:,:,2] = red/256
    hist_b = cv2.calcHist([img], [0], None, [dim], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [dim], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [dim], [0, 256])
    hist_r[-5:] = np.ceil(hist_r[-5:]/10)
    return hist_b, hist_g, hist_r


# In[14]:


def get_hist_data(img_organs, dim):
    organ_hist = {}
    for i, organ in enumerate(img_organs):
        img = img_organs[organ]
        hist_b, hist_g, hist_r = get_hist(img, dim)
        msg_json = []
        for i in range(hist_b.shape[0]):
            msg_json.append({'dim':[i,i+1],
                            'blue':int(hist_b[i][0]),
                            'green':int(hist_g[i][0]),
                            'red':int(hist_r[i][0]),
                            'dims':i,
                           })
        organ_hist[organ] = msg_json
    return organ_hist


# In[15]:


def get_organ_feature(dim, img_organs, chosen_organs):
    sample_num = 1
    bgr_data = np.zeros( ( sample_num, dim*3  ) )

    organ_feat = {}
    for organ in chosen_organs:
        img = img_organs[organ]
        hist_b, hist_g, hist_r = get_hist(img, dim)
        organ_feat[organ] = np.concatenate([hist_b, hist_g, hist_r]).T
        
    return organ_feat


# ### get result

# In[16]:


def predict_result(organ_feat, img_organs, color_labels, face_diagnose, landmark):
    predicts = []
    results = []
    res_imgs = []
    
    predict_organ = {}
    for organ in organ_feat:
        res_imgs.append(img_organs[organ])
        try:
            modle_path = model_dir + '/svm_'+organ+'_model.m'
            model = joblib.load(modle_path)
            if debug:
                print (modle_path)
            # predict data
            predict = model.predict(organ_feat[organ])

            predict_organ[organ]=predict[0] #save organ predict
            predicts.append(color_labels[predict[0]])
            results.append(organ+': '+color_labels[predict[0]])
        except Exception:
            pass
        
        color_diagnose = Counter(predicts).most_common(1)
    if debug:
        grid_display(res_imgs, results, 5, (15,15), 1, conv_color = True)
        
#         grid_display(img_origs, color_diagnose, 1, (10,10), 1, conv_color = True)
        
        # print (color_diagnose[0][0])
        print (face_diagnose[color_diagnose[0][0]])

        # process eyes
        grid_display([img_organs['eyes']], ['eyes'], 1, (8,3), 1, conv_color = True)
        # classify smart or not
        if (landmark[0][36][1] < landmark[0][39][1] or landmark[0][42][1] > landmark[0][45][1]):
            print('双目有神')
        else:
            print('双目无神')

        # process lip
        grid_display([img_organs['lip']], ['lip'], 1, (3,2), 1, conv_color = True)
        # classify smart or not
        
    return predict_organ, color_diagnose[0][0]


# ### generate donut data

# In[17]:


def generate_rand_list():
    import random
    rand_list = []
    sum_rand = 0
    random.randint(1, 100)
    for i in range(6):
        rand = random.randint(1, 100)
        rand_list.append(rand)
        
        sum_rand += rand

    color_prob = []
    for i, num in enumerate(rand_list):
        prob = num / sum_rand * 60
        
        color_prob.append(prob)
    
    
    return color_prob


def generate_res_prob(predict_organ, color_type):
    color = []
    if color_type == 'face':
        color = ['黑','青','黄','白','赤', '正常']
    else:
        color = ['淡红','淡白','红','绛','青紫', '正常']
    
    res_prob = []
    reulst_prob = {}

    for organ in predict_organ:
        color_prob = generate_rand_list()
        res_prob.append(color_prob)
        sum_s = 0
        color_res_prob = []
        for i, count in enumerate(color_prob):
            color_res_prob.append({'item': color[i], 'count':round(count)})
            if i != predict_organ[organ]:
                sum_s += round(count)
            
        color_res_prob[ predict_organ[organ] ] = {'item': color[ predict_organ[organ] ], 'count':100 - sum_s}
        reulst_prob[organ] = [color[predict_organ[organ]] , color_res_prob]
   
    return reulst_prob

# reulst_prob =  generate_res_prob(predict_organ)
# print(reulst_prob)


# ### generater organ results

# In[18]:


def get_organ_result(organ_results, predict_organ):
    organ_res = {}
    for key in predict_organ:
        organ_res[key] = organ_results[key][predict_organ[key]]
    return organ_res
#     print (organ_res)


# ### get polar data

# In[19]:


def get_ploar_data(predict_organ, organ_chosen, color_type):
    init_num = 10
    color_result = []
    reorder_Res = {0:3,1:4,2:1,3:2,4:0}
    oneColorRes = []
    key = ''
    if  color_type == 'face':
        key_type = '面色'
        oneColorRes = [{ 'item': '心/赤', '面色': init_num }, { 'item': '脾/黄', '面色': init_num }, { 'item': '肺/白', '面色': init_num }, 
                   { 'item': '肾/黑', '面色': init_num }, { 'item': '肝/青', '面色': init_num },]
    else:
        key_type = '舌色'
        oneColorRes = [{ 'item': '淡白', '舌色': init_num }, { 'item': '赤', '舌色': init_num }, { 'item': '淡红', '舌色': init_num },  
                { 'item': '淡紫', '舌色': init_num }, { 'item': '绛', '舌色': init_num }, ]
    
    for i, key in enumerate(organ_chosen):
        oneColorRes[ reorder_Res[ predict_organ[key] ] ][key_type] += 6
        oneColorRes[i][key_type] += 10
        color_result.append(copy.deepcopy(oneColorRes))
    return color_result
# print (color_result)


# ### download image/ upload result

# In[20]:


def query_patient_basic_info(lc_class, col_name, val_name, col_id, val_id, col_bir_date, col_gender, today):
    #query leancloud
    Query = leancloud.Object.extend(lc_class)
    query1 = Query.query
    query2 = Query.query
    query1.equal_to(col_name, val_name)
    query2.equal_to(col_id, val_id)
    # join query
    query = leancloud.Query.and_(query1,query2)
    age = None
    gender = None
    try: 
        req_res = query.first()
        bd = req_res.get(col_bir_date), 
        gender = req_res.get(col_gender)
#         print (f'bd {bd}')
        age = int(today.split('-')[0]) - int(bd[0].split('-')[0])
    except Exception as error:
        print(repr(error))
        print('query 0 basic patient info object!')
    return age, gender

def zonetime_gtm(timezone): 
    tz = pytz.timezone(timezone) 
    now = datetime.datetime.now(tz) 
    timezone_str = now.strftime('%Y-%m-%d')
    return timezone_str 

def get_leancloud_url(lc_class, col_name, val_name, col_id, val_id, col_date, val_date, col_furl, col_turl, col_obj_id):
    #query leancloud
    Query = leancloud.Object.extend(lc_class)
    query1 = Query.query
    query2 = Query.query
    query3 = Query.query
    query1.equal_to(col_name, val_name)
    query2.equal_to(col_id, val_id)
    query3.equal_to(col_date, val_date)
    # join query
    query = leancloud.Query.and_(query1,query2,query3)
#     help(query_result)
    # check patient whether take photo
    furl = None
    turl = None
    obj_id = None
    try:
        req_res = query.first()
        return req_res.get(col_furl), req_res.get(col_turl), req_res.get(col_obj_id)
    except Exception as error:
        print(repr(error))
        print('query 0 patient photographed info!')
        return None, None, None

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def save_image(image, path, fx, fy):
    image = cv2.resize(image, (0,0), fx=fx, fy=fy)
    cv2.imwrite(path, image)

def update_result(lc_class, obj_id, col_fres, col_tres, face_result, tongue_result):
    if obj_id != 'null':
        update_res = leancloud.Object.extend(lc_class)
        update = update_res.create_without_data(obj_id)
        # 这里修改 location 的值
        update.set(col_fres, face_result)
        update.set(col_tres, tongue_result)
        update.save()
        print("update successfully")
    else:
        print("wrong obj_id, update fail")
    
def download_input_image(pf_url, pt_url, obj_id, return_dict):
    if debug:
        print (pf_url)
        print (pt_url)
        print (obj_id)
    dl_img = url_to_image(pf_url)
    save_image(dl_img, input_face_path, 1, 1)
    print ("wrote image {}".format(input_face_path))
    dl_img = url_to_image(pt_url)
    save_image(dl_img, input_tongue_path, 1, 1)
    print ("wrote image {}".format(input_tongue_path))
    print (f'obj_id {obj_id}' )
    return_dict[0] = obj_id
    
def set_timeout_download(pf_url, pt_url, obj_id, timeout):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict[0] = 'null'
    p = multiprocessing.Process(target=download_input_image, name="download_input_image", args=(pf_url, pt_url, obj_id, return_dict))
    p.start()
    # main thread keep checking sub thread result
    for i in range(timeout*5):
        if(i%5 == 0):
            print (f"return_dict[0] {return_dict[0]}")
        if return_dict[0] != 'null':
            p.terminate()
            print (f"return_dict[0] {return_dict[0]}")
            return return_dict[0]
        time.sleep(0.2)
    # thread is active
    print (f"download is running for to long, let's kill it...")
    p.terminate()
    p.join()
    return return_dict[0]

# res = query_patient_basic_info('patient_info', 'name', '张自忠', 'cell', '123123')
# print (res)


# In[22]:


class leancloud_jobs():
    def __init__(self):
        appid = leancloudid
        appkey = leancloudkey
        leancloud.init(appid, appkey)
        self.object_id = 'null'
        self.age_category = 0
        self.gender = 0
    def query_register_info(self, lc_class, col_name, val_name, col_id, val_id, col_bd, col_gender, year):
        age, gender =  query_patient_basic_info(lc_class, col_name, val_name, col_id, val_id, col_bd, col_gender, year)
        if age and gender:
            print (f'age {age}')
            if age < 40: self.age_category = '0'
            elif age < 55: self.age_category = '1'
            else: self.age_category = '2'
            self.gender = str(gender)
            return True
        else:
            return False
    def get_date(self, timezone):
        na_date = zonetime_gtm(timezone)
        return na_date
    def query_pic_url(self, lc_class, col_name, val_name, col_id, val_id, col_date, val_date, col_furl, col_turl, col_obj_id):
        self.pf_url, self.pt_url, self.obj_id = get_leancloud_url(
            lc_class, col_name, val_name, col_id, val_id, col_date, val_date, col_furl, col_turl, col_obj_id)
        print (self.pf_url)
        print (self.pt_url)
        print (self.obj_id)
        if self.pf_url and self.pt_url and self.obj_id:
            return True
        else:
            return False
    def download_with_timeout(self, timeout):
#         print (self.pf_url)
#         print (self.pt_url)
#         print (self.obj_id)
        self.object_id = 'null'
        self.object_id = set_timeout_download(self.pf_url, self.pt_url, self.obj_id, timeout)
        if self.object_id == 'null':
            return False
        else:
            return True
    def update_results(self, lc_class, col_fres, col_tres, face_result, tongue_result):
        update_result(lc_class, self.object_id, col_fres, col_tres, face_result, tongue_result)
        
if notebook:
    lc = leancloud_jobs()
    na_date = lc.get_date('Canada/Pacific')
    is_register = lc.query_register_info('patient_info', 'name', '郭某', 'cell', '123', 'birthDate', 'gender', na_date)
    print (f'is_register {is_register} age {lc.age_category} gender {lc.gender}')
    #     'Asia/Shanghai' 'America/Chicago'
    print (f'na_date {na_date}')
    photographed = lc.query_pic_url('patient_current_info', 'name', '小白', 'cell', '123', 'currentdate', na_date, 'facialpictureURL', 'tonguepictureURL', 'objectId')
    print (f'photographed {photographed}')
    downloaded = lc.download_with_timeout(20)
    print (f'downloaded {downloaded}')
    lc.update_results('patient_current_info', 'facialdiagnosiscomputer', 
                  'tonguediagnosiscomputer', fdd.face_res[0].split('：')[0], tdd.face_res[0].split('：')[0])


# In[33]:


class diagnose_face_data():
    def __init__(self):
        # self.organ_chosen = ['jaw', 'cheek_l', 'cheek_r', 'nose', 'lip', 'forehead',  'neck', 'tongue', ]
        self.organ_chosen = ['forehead', 'cheek_l', 'nose', 'cheek_r', 'jaw']
        self.color_labels = ['black', 'blue', 'yellow', 'white', 'red']
        self.organ_dirs = {'jaw':'./sample_sets/jaw/', 
              'cheek_l':'./sample_sets/cheek_l/', 
              'cheek_r':'./sample_sets/cheek_r/', 
              'nose':'./sample_sets/nose/', 
              'lip':'./sample_sets/lip/', 
              'eyes':'./sample_sets/eyes/',
              'forehead':'./sample_sets/forehead/', 
#               'neck':'./sample_sets/neck/',
              'tongue':'./sample_sets/tongue/'
             }
        with open(json_dir+'facial_res.json', 'r') as fp:
            self.face_diagnose = json.load(fp)
#             print(self.face_diagnose)
#         self.face_diagnose = {
#             'red': ['面赤色：\n面色赤色为暑热之色，手少阴经之色，心包络，小肠之色。主热证，赤色重为实热，微赤为虚热。因气血得热则行，热盛而血脉充盈，血色上荣，所以面色赤红。',\
#                    '健康建议：\n红色对应心，面色红多与心有关。推荐几种养心的食物，如苦菜，大头菜，白果等都是很好的养心食物。' ],
#             'yellow': ['面黄色：\n面部黄色为湿土之色、脾胃之色、足太阴经之色。为风为热，主虚证、湿证。黄色乃脾虚湿蕴之象征。脾失健运、水湿内停、气血不充，致使肌肤失于充养，所以面色发黄。',\
#                        '健康建议：\n黄色和脾对应，面色黄多与脾有关。脾为气血生化之源。脾胃功能运健，则气血旺盛，见面色红润，肌肤弹性良好。下面推荐几种养脾是食物，如茄子、蘑菇、胡萝卜、土豆、黄瓜、冬瓜、藕、梨、苹果、香蕉、西瓜。'],
#             'white':['面白色：\n面部白色为燥金之色，手太阴经之色，肺与大肠之色。主寒证、虚证、脱血、夺气。白色为气血虚弱不能荣养机体的表现。',\
#                      '健康建议：\n阳气不足，气血运行无力，或耗气失血不充，血脉空虚，均可呈现白色。面白对应肺，面色白多与肺有关。肺的气机以宣降为顺，人体通过肺气的宣发和肃降，使气血津液得以布散全身。这里推荐几种养肺的食物，如胡椒、辣椒、葱、蒜、花椒等都是很好的养肺食物 '],
#             'black':['面黑色：\n面黑色主肾虚证、水饮证、寒证、痛证及淤血证。黑为阴寒水盛之色。',\
#                      '健康建议：\n由于肾阳虚衰，水饮不化，气化不行，阴寒内盛，血失温养，经脉拘急，气血不畅，故面色黧黑。黑色对应肾，面色黑多与肾有关。肾主藏精。肾精充盈，肾气旺盛时，五脏功能方可正常运行。推荐几种养肾食物，如海带、紫菜、海参等都是很好的养肾食物。'],
#             'blue':['面青色：\n面青色主寒证、痛证、淤血证、惊风证、肝病。青色为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉，则气滞血瘀，故面色发青。',\
#                     '健康建议：\n青色与肝对应，面色青多与肝有关。肝主藏血，主疏泄，能调节血流量和调畅全身气机，使气血平和，面部血液运行充足。养肝的食物有橘子、橄榄、柠檬、枇杷、芒果、石榴等。'],
#             }
        self.organ_results = {'jaw': [["下巴偏黑", "主肾虚证、水饮证、寒证、痛证及淤血证。为阴寒水盛之色。"], ["下巴偏黑", "主肾虚证、水饮证、寒证、痛证及淤血证。为阴寒水盛之色。"],["下巴偏黑", "主肾虚证、水饮证、寒证、痛证及淤血证。为阴寒水盛之色。"],["下巴偏黑", "主肾虚证、水饮证、寒证、痛证及淤血证。为阴寒水盛之色。"],["下巴偏黑", "主肾虚证、水饮证、寒证、痛证及淤血证。为阴寒水盛之色。"],],
             'cheek_l': [["左脸偏青", "为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉。"],["左脸偏青", "为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉。"],["左脸偏青", "为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉。"],["左脸偏青", "为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉。"],["左脸偏青", "为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉。"],],
             'cheek_r':[["右脸偏白", "乃气血虚弱不能荣养机体的表现。"],["右脸偏白", "乃气血虚弱不能荣养机体的表现。"],["右脸偏白", "乃气血虚弱不能荣养机体的表现。"],["右脸偏白", "乃气血虚弱不能荣养机体的表现。"],["右脸偏白", "乃气血虚弱不能荣养机体的表现。"],], 
             'nose':[["鼻偏黄", "脾虚湿蕴之象征。脾失健运、水湿内停、气血不充。"],["鼻偏黄", "脾虚湿蕴之象征。脾失健运、水湿内停、气血不充。"],["鼻偏黄", "脾虚湿蕴之象征。脾失健运、水湿内停、气血不充。"],["鼻偏黄", "脾虚湿蕴之象征。脾失健运、水湿内停、气血不充。"],["鼻偏黄", "脾虚湿蕴之象征。脾失健运、水湿内停、气血不充。"],],
             'forehead':[["额偏赤", "两颧潮红，如指如褛者，属阴虚证。"],["额偏赤", "两颧潮红，如指如褛者，属阴虚证。"],["额偏赤", "两颧潮红，如指如褛者，属阴虚证。"],["额偏赤", "两颧潮红，如指如褛者，属阴虚证。"],["额偏赤", "两颧潮红，如指如褛者，属阴虚证。"]],
             'tongue':[["舌淡红", "气血得热则行，热盛而血脉充盈，血色上荣"],["舌淡红", "气血得热则行，热盛而血脉充盈，血色上荣"],["舌淡红", "气血得热则行，热盛而血脉充盈，血色上荣"],["舌淡红", "气血得热则行，热盛而血脉充盈，血色上荣"],["舌淡红", "气血得热则行，热盛而血脉充盈，血色上荣"],]
                }
        appid = leancloudid #leandcloud id
        appkey = leancloudkey #leandcloud key
        leancloud.init(appid, appkey)
        self.input_path = input_face_path #input picture
        self.save_resize = (0.5,0.5) #save download picture with resized size (w_rate, h_rate)
        self.data_dim = 64
        self.actual_dim =256
        self.organ_resize = (120, 120)
        self.img_organs = {}
        self.organ_boxes = {}
        self.landmark = []
        self.img_shape = ()
        self.organ_position = []
        self.hist_data = {} #hist_chart_data
        self.hist_feat = {} #hist_feature
        self.predict_organ = {} #predict num of each organ
        self.reulst_prob = {} #donut_data
        self.organ_res = {} #diagnose of each organ
        self.polar_res = [] #result polar data 
        self.face_res_color = ''
        self.face_res = ['',''] #face diagnose result
        self.model_dlib = dlib.shape_predictor(dlib_path)
    def download_input_image(self, lc_class, column, value, column_out):
#         download_input_image(self.appid, self.appkey, '_File', '123450')
        url = get_leancloud_url(lc_class, column, value, column_out)
        print (url)
        dl_img = url_to_image(url)
        save_image(dl_img, self.input_path, self.save_resize[0], self.save_resize[1])
        print ("wrote image {}".format(self.input_path))
    def collect_data(self):
        self.img_organs, self.organ_boxes, self.img_shape, self.landmark = detect_face(self.input_path, self.organ_dirs, self.organ_resize, self.model_dlib)
        self.organ_position = get_organ_position(self.organ_boxes, self.img_shape, -1)
#         print (f'self.organ_boxes {self.organ_boxes}')
#         print (self.img_organs.keys())
#         print (self.organ_position)
        self.hist_data = get_hist_data(self.img_organs, self.data_dim)
#         print (self.hist_data['nose'][20:30])
        self.hist_feat = get_organ_feature(self.actual_dim, self.img_organs, self.organ_chosen)
#         print (self.hist_feat['nose'].shape)
#         print (self.img_organs.keys())
    def collect_reult(self, cat_age, cat_gender, set_face=''):
        self.predict_organ, self.face_res_color = predict_result(self.hist_feat, self.img_organs, self.color_labels, self.face_diagnose, self.landmark)
#         print(self.face_res_color)
        if self.face_res_color == 'black':
            self.face_res_color = 'blue'
        if set_face and set_face != 'normal':
            self.face_res = self.face_diagnose[set_face][cat_gender][cat_age]
        else:
            self.face_res = self.face_diagnose[self.face_res_color][cat_gender][cat_age]
#         print (self.predict_organ)
        print (self.face_res[0])
#         print (self.face_res[1])
        self.reulst_prob = generate_res_prob(self.predict_organ, 'face')
#         print(self.reulst_prob) 
        self.organ_res = get_organ_result(self.organ_results, self.predict_organ)
#         print (self.organ_res)
#         print (self.predict_organ)
#         print (self.organ_chosen)
        self.polar_res = get_ploar_data(self.predict_organ, self.organ_chosen, 'face')
#         print (self.polar_res)

if notebook:
#     try:
#         fdd = diagnose_face_data() #face_diagnose_data
#     #     fdd.download_input_image('patient_current_info', 'patientID', '1022', 'facialpictureURL')
#         fdd.collect_data()
#         fdd.collect_reult()
#     except Exception as error:
#         print(repr(error))
        
    fdd = diagnose_face_data() #face_diagnose_data
#     fdd.download_input_image('patient_current_info', 'patientID', '1022', 'facialpictureURL')
    fdd.collect_data()
    
    fdd.collect_reult(lc.age_category, lc.gender )


# In[29]:


class diagnose_tongue_data():
    def __init__(self):
        # self.organ_chosen = ['jaw', 'cheek_l', 'cheek_r', 'nose', 'lip', 'forehead',  'neck', 'tongue', ]
        self.organ_chosen = ['tongue']
        self.color_labels = ['black', 'blue', 'yellow', 'white', 'red']
        self.organ_dirs = {
              'tongue':'./sample_sets/tongue/'
             }
        with open(json_dir+'tongue_res.json', 'r') as fp:
            self.tongue_diagnose = json.load(fp)
#         self.tongue_diagnose = {
#             0: ['舌淡紫：\n胃肾阴伤，热久伤阴，阴虚水涸，虚火上炎，属虚热征。',\
#                    '健康建议：\n红色对应心，红多与心有关。推荐几种养心的食物，如苦菜，大头菜，白果等都是很好的养心食物。' ],
#             1: ['舌绛：\n面部黄色为湿土之色、脾胃之色、足太阴经之色。为风为热，主虚证、湿证。黄色乃脾虚湿蕴之象征。脾失健运、水湿内停、气血不充，致使肌肤失于充养，所以面色发黄。',\
#                    '健康建议：\n黄色和脾对应，面色黄多与脾有关。脾为气血生化之源。脾胃功能运健，则气血旺盛，见面色红润，肌肤弹性良好。下面推荐几种养脾是食物，如茄子、蘑菇、胡萝卜、土豆、黄瓜、冬瓜、藕、梨、苹果、香蕉、西瓜。'],
#             2:['舌赤：\n气血不足，阳气不足。阳虚内寒，筋络收引,故气血不能上荣于舌。',\
#                      '健康建议：\n阳气不足，气血运行无力，或耗气失血不充，血脉空虚，均可呈现白色。面白对应肺，面色白多与肺有关。肺的气机以宣降为顺，人体通过肺气的宣发和肃降，使气血津液得以布散全身。这里推荐几种养肺的食物，如胡椒、辣椒、葱、蒜、花椒等都是很好的养肺食物 '],
#             3:['舌淡红：\n面黑色主肾虚证、水饮证、寒证、痛证及淤血证。黑为阴寒水盛之色。',\
#                      '健康建议：\n由于肾阳虚衰，水饮不化，气化不行，阴寒内盛，血失温养，经脉拘急，气血不畅，故面色黧黑。黑色对应肾，面色黑多与肾有关。肾主藏精。肾精充盈，肾气旺盛时，五脏功能方可正常运行。推荐几种养肾食物，如海带、紫菜、海参等都是很好的养肾食物。'],
#             4:['舌淡白：\n面青色主寒证、痛证、淤血证、惊风证、肝病。青色为筋脉阻滞，气血不通之象。寒主收引主凝滞，寒盛而留于血脉，则气滞血瘀，故面色发青。',\
#                     '健康建议：\n青色与肝对应，面色青多与肝有关。肝主藏血，主疏泄，能调节血流量和调畅全身气机，使气血平和，面部血液运行充足。养肝的食物有橘子、橄榄、柠檬、枇杷、芒果、石榴等。'],
#             }
        self.organ_results = {'tongue':[["舌淡紫", "体质偏虚寒，冬春较易感冒，秋冬易咳嗽"],                                        ["舌绛", "虚火上升，情绪易波动，多梦易醒"],                                        ["舌赤", "阴虚火旺，津液亏少，口燥咽干"],                                        ["舌淡红", "阴阳平衡，未出现明显偏颇"],                                        ["舌淡白", "气血较虚，时有疲乏，睡眠不实"],]
                }
        appid = leancloudid #leandcloud id
        appkey = leancloudkey #leandcloud key
        leancloud.init(appid, appkey)
        self.input_path = input_tongue_path #input picture
        self.save_resize = (0.5,0.5) #save download picture with resized size (w_rate, h_rate)
        self.data_dim = 64
        self.actual_dim =256
        self.organ_resize = (120, 120)
        self.img_organs = {}
        self.organ_boxes = {}
        self.landmark = []
        self.img_shape = ()
        self.organ_position = []
        self.hist_data = {} #hist_chart_data
        self.hist_feat = {} #hist_feature
        self.predict_organ = {} #predict num of each organ
        self.reulst_prob = {} #donut_data
        self.organ_res = {} #diagnose of each organ
        self.polar_res = [] #result polar data 
        self.face_res_color = 0
        self.face_res = ['',''] #face diagnose result
        self.model_dlib = dlib.shape_predictor(dlib_path)
    def download_input_image(self, lc_class, column, value, column_out):
#         download_input_image(self.appid, self.appkey, '_File', '123450')
        url = get_leancloud_url(lc_class, column, value, column_out)
        print (url)
        dl_img = url_to_image(url)
        save_image(dl_img, self.input_path, self.save_resize[0], self.save_resize[1])
        print ("wrote image {}".format(self.input_path))
    def collect_data(self):
        self.img_organs, self.organ_boxes, self.img_shape, self.landmark = detect_face(self.input_path, self.organ_dirs, self.organ_resize, self.model_dlib)
        self.organ_position = get_organ_position(self.organ_boxes, self.img_shape, 1)
#         print (self.img_organs.keys())
#         print (self.organ_position)
        self.hist_data = get_hist_data(self.img_organs, self.data_dim)
#         print (self.hist_data)
        self.hist_feat = get_organ_feature(self.actual_dim, self.img_organs, self.organ_chosen)
#         print (self.hist_feat['nose'].shape)
#         print (self.img_organs.keys())
    def collect_reult(self, cat_age, cat_gender, set_tongue=''):
#         self.predict_organ, self.face_res_color = predict_result(self.hist_feat, self.img_organs, self.color_labels, self.tongue_diagnose, self.landmark)
        img_orig = cv2.imread(tongue_pic_path)
        self.face_res_color, img_after, img_hist0, tongue_edge_feat = tgd.diagnose_tongue(img_orig, tongue_mask_path)#diagnose_tongue(self.img_organs['tongue'])
#         plt.plot(range(180), img_hist0, label=tongue_edge_feat)
#         plt.show()
#         print (f'self.face_res_color', self.face_res_color)
#         grid_display([img_after], list_of_titles=[self.face_res_color], no_of_columns=1, figsize=(5,5), ratio=1, conv_color = True)
        if set_tongue:
            self.face_res_color = int(set_tongue)
            
        self.face_res = self.tongue_diagnose[str(self.face_res_color)][cat_gender][cat_age]   
#         print (self.predict_organ)
        print (self.face_res[0])
    
        self.predict_organ = {'tongue': self.face_res_color}
        self.reulst_prob = generate_res_prob(self.predict_organ, 'tongue')
#         print(self.reulst_prob)
        self.organ_res = get_organ_result(self.organ_results, self.predict_organ)
        # print (self.organ_res)
#         self.organ_chosen = ['kidney', 'livgb', 'splsto', 'livgb', 'htlg']
        temp_predict_organ = {'kidney': self.face_res_color, 'livlgb': self.face_res_color, 
                              'splsto': self.face_res_color, 'livrgb': self.face_res_color, 'htlg': self.face_res_color}
        self.polar_res = get_ploar_data(temp_predict_organ, ['kidney', 'livlgb', 'splsto', 'livrgb', 'htlg'], 'tongue')
        # print (self.polar_res)

if notebook:
#     try:
#         tdd = diagnose_tongue_data()
#     #     tdd.download_input_image('patient_current_info', 'patientID', '1022', 'tonguepictureURL')
#         tdd.collect_data()
#         tdd.collect_reult()
#     except Exception as error:
#         print(repr(error))
    tdd = diagnose_tongue_data()
#     tdd.download_input_image('patient_current_info', 'patientID', '1022', 'tonguepictureURL')
    tdd.collect_data()
    tdd.collect_reult(lc.age_category, lc.gender, '2')


# In[ ]:




