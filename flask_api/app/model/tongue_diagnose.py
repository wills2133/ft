# coding: utf-8

# In[1]:


import __future__
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import copy


# In[2]:


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


# In[3]:


def find_boder_valley(hist, max_loc, total):
    index = max_loc-2
    border = 3
    p1=0
    p2=0
    v0=0
    while index > 3:

        p1 = hist[index]
        v0 = hist[index+1]
        p2 = hist[index+2]
        # print (p1)
        # print (p2)
        # print (v0)
        if p1>v0 and p2>v0:
            border =  index+1
            break
        index -= 1
    hist_sum = 0
    for i in range(150, 179):
        hist_sum += hist[i]
    for i in range(border):
        hist_sum += hist[i]
    limit = 0.4*total
#     print ("sum0={}, boder0={}".format(hist_sum, border))
    # print (p1)
    # print (p2)
    # print (v0)
#     print (limit)
    while  ( hist_sum < limit ) or not ( p1>v0 and p2>v0 ) :
#         print ("sum={}, boder={}".format(hist_sum, border))
        border += 1
        p1 = hist[border-1]
        v0 = hist[border]
        p2 = hist[border+1]
        hist_sum += hist[border]
    return border


# In[4]:


def find_max_area(mask):
    mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, 1)
    #show the contours of the imput image
    #find the max area of all the contours and fill it with 0
    mask_max_area = np.zeros(mask.shape, mask.dtype)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    all_y0=[]
    contours_cut=[]
    contour_res = contours[max_idx]
    # connect upper border
    for i, coord in enumerate(contours[max_idx]):
        if coord[0][1] == 0:
            all_y0.append(i)
    if len(all_y0) > 10:
        for i, coord in enumerate(contours[max_idx]):
            if i < all_y0[1] or i > all_y0[-1]:
                contours_cut.append([ coord[0] ])
        contour_res = np.array(contours_cut)
    cv2.fillConvexPoly(mask_max_area, contour_res, 255)
    return mask_max_area, mask, contour_res


# In[5]:


def cal_center(contour_res):
    h_sum = 0
    w_sum = 0
    n = contour_res.shape[0]
    for p in contour_res:
        h_sum += p[0][0]
        w_sum += p[0][1]
    width = contour_res[:,:,0].max() - contour_res[:,:,0].min()
#     print (contour_res[:,:,0].max())
#     print (contour_res[:,:,0].min())
    center = (int(w_sum/n), int(h_sum/n))
    return center, width


# In[6]:


def Gaussian_Blur(img, size):
    if size > 0:
        size = size+size-1
        # Gaussian denoising
        size = size+size+1
        blurred = cv2.GaussianBlur(img, (size, size),0)

        return blurred
    else:
        return img


# In[7]:


def image_morphology(img, size):
    if size > 0:
        size = size+size-1
        # 建立一个椭圆核函数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # 执行图像形态学, 细节直接查文档，很简单
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#         closed = cv2.erode(closed, kernel, iterations=4)
#         closed = cv2.dilate(closed, None, iterations=4)
        return closed
    else:
        return img


# In[8]:


def collect_edge(mask, size):
    mask_erode = copy.copy(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # 执行图像形态学, 细节直接查文档，很简单
    mask_erode = cv2.erode(mask_erode, kernel, iterations=4)
    mask_erode = cv2.erode(mask_erode, None, iterations=4)
    mask_erode = cv2.dilate(mask_erode, None, iterations=4)
    mask_erode = cv2.bitwise_not(mask_erode)
    mask_erode = cv2.bitwise_and(mask, mask_erode)
    return mask_erode


# In[16]:


def save_png(mask, color, path):
    # creat a blank image
    height = mask.shape[0]
    width = mask.shape[1]
    blank_image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(color))
    blank_image[:] = color
    # save tongue mask png
    b, g, r = cv2.split(blank_image)
    rgba = [b,g,r, mask]
    dst = cv2.merge(rgba,4)
    cv2.imwrite(path, dst)
    


# In[10]:


def filter_bottom(mask, contour, rate):
    top = contour[:,:,1].min()
    bottom = contour[:,:,1].max()
#     print ("top={}, bottom={}".format(top, mask.shape[0]))
#     print ("top={}, bottom={}".format(top, bottom))
    #calculate how much to be remove
    remove = int( (bottom - top) * rate )
    cv2.rectangle(mask, (0,0), (mask.shape[1], top+remove), 0, -1) #-1 means fill with value 0
    return mask


# In[11]:


def collect_value(hist):
    Integral = 0
    for i, val in enumerate(hist):
        Integral += i * val
    return Integral


# In[12]:


def get_hsv_hist(tongue_edge):
    img_hist0 = cv2.calcHist([tongue_edge], [0], None, [180], [0, 179])
    #normalize
    img_hist0[0] = 0
    hist_sum = img_hist0.sum()
    img_hist0 = img_hist0 * 100 / hist_sum
    return img_hist0


# In[13]:


def get_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ### Now predict

# In[21]:


def diagnose_tongue(img_orig, tongue_mask_path):
    blur = 6
    H_up = 150
    #transform to HSV
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    #get hist
    img_hist = cv2.calcHist([img], [0], None, [180], [0, 179])
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_hist[3:150])
    #     print (maxLoc)
    H_lo = find_boder_valley(img_hist, maxLoc[1], img.shape[0]*img.shape[1])
    #     print (H_lo)
    #set threshold
    lower = np.array([H_lo,0,0])
    upper = np.array([H_up,255,255])
    # extract mask according to the threshold
    mask = cv2.inRange(img, lower, upper)
    mask = cv2.bitwise_not(mask)
    # morphology process
    mask = image_morphology(mask, blur)
    mask_max_area, mask, contour_res = find_max_area(mask)
    
    # collect tongue edge
    mask_edge = collect_edge(mask_max_area, 22)    
    mask_side_edge = filter_bottom(mask_edge, contour_res, 0.3)
    # draw tongue contour
    #     cv2.drawContours(img_orig, [contour_res], -1, (0, 255, 255), 2)
    # extrack edge of original image

    tongue_edge = cv2.bitwise_and(img_orig, img_orig, mask=mask_side_edge)
    #down sampling
    #     tongue_edge = cv2.resize(tongue_edge, (0,0), fx=0.1, fy=0.1)
    #     tongue_edge = cv2.resize(tongue_edge, (0,0), fx=10, fy=10)
    #     image = cv2.resize(image, (0,0), fx=fx, fy=fy)
    #     img_orig = cv2.resize(img_orig, (0,0), fx=5, fy=5)
    img_hist0 = get_hsv_hist(tongue_edge)
    tongue_edge_feat = collect_value(img_hist0)  

    tongue_pred = -1
    if tongue_edge_feat<9000: tongue_pred = 0
    elif tongue_edge_feat<10000: tongue_pred = 1
    elif tongue_edge_feat<12000: tongue_pred = 2
    elif tongue_edge_feat<13000: tongue_pred = 3
    elif tongue_edge_feat<15000: tongue_pred = 4
    # save png
    save_png(mask_max_area, (255, 255, 255), tongue_mask_path)
    return tongue_pred, tongue_edge, img_hist0, tongue_edge_feat