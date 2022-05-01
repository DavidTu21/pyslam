import sys 
import numpy as np
import cv2 
from matplotlib import pyplot as plt

sys.path.append("/Users/mingxiaotu/Desktop/pyslam/")
from config import Config

from mplot_figure import MPlotFigure
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from utils_img import combine_images_horizontally, rotate_img, transform_img, add_background
from utils_geom import add_ones
from utils_features import descriptor_sigma_mad
from utils_draw import draw_feature_matches

from feature_tracker_configs import FeatureTrackerConfigs

from timer import TimerFps

# cd to the parent folder
# Activate the environment $ . pyenv-activate.sh 
# python3 test/cv/test_feature_matching.py 1

def combine_images_horizontally(img1, img2): 
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)    
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)                     
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[:h2, w1:w1+w2,:3] = img2
    return img3 

def draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_horizontally(img1,img2)    
    h1,w1 = img1.shape[:2]    
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size   
    i = 0     
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i] 
        size2 = kps2_sizes[i]    
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)    # optic flow style         
        cv2.line(img3, (a,b),(c+w1,d), color, 1)  # join corrisponding points 
        cv2.circle(img3,(a,b),2, color,-1)   
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=int(size1), thickness=1)  # draw keypoint size as a circle 
        cv2.circle(img3,(c+w1,d),2, color,-1) 
        cv2.circle(img3,(c+w1,d), color=(0, 255, 0), radius=int(size2), thickness=1)  # draw keypoint size as a circle  
        i += 1
    print(i)
    return img3
# Compute homography reprojection error
def compute_hom_reprojection_error(H, kps1, kps2, img2, mask=None):   

    # @ is a binary operator, used for matrix multiplicaiton
    kps1_reproj = H @ add_ones(kps1).T
    
    #print(f'kps1_reproj has a shape of {np.shape(kps1_reproj)}') # (3,14)
    kps1_reproj = kps1_reproj[:2]/kps1_reproj[2]
    prediction = kps1_reproj.T
    # for i in range(len(prediction)):
    #     # red is the predicted
    #     cv2.circle(img2, (int(prediction[i][0]),int(prediction[i][1])), 20, (0,0,255), 2)
    #print(f'kps1_reproj has a shape of {np.shape(kps1_reproj)}') # (2,14)
    error_vecs = kps1_reproj.T - kps2
    dis_ls = []
    
    # calculate the average distance, they are tuples btw
    for i in range(len(kps2)):
        #print(i,kps2[i][0],kps2[i][1], prediction[i][0], prediction[i][1])
        # sqrt((x2-x1)^2 + (y2-y1)^2)
        dis = ((kps2[i][0] - prediction[i][0])**2 + (kps2[i][1] - prediction[i][1])**2)**0.5
        dis_ls.append(dis) 
    print(f'The value returned by using Euclean distance is {np.mean(dis_ls)}, the standard deviation is {np.std(dis_ls)}')
    
    #return np.mean(np.sum(error_vecs*error_vecs,axis=1))
    return
# ==================================================================================================
# N.B.: test the feature tracker and its feature matching capability 
# ==================================================================================================

timer = TimerFps(name='detection+description+matching')


#============================================
# Select Images   
#============================================  

img1, img2 = None, None       # var initialization
img1_box = None               # image 1 bounding box (initialization)
model_fitting_type = None     # 'homography' or 'fundamental' (automatically set below, this is an initialization)
draw_horizontal_layout=True   # draw matches with the two images in an horizontal or vertical layout (automatically set below, this is an initialization) 

test_type='rat'             # select the test type (there's a template below to add your test)
#  
if test_type == 'box': 
    img1 = cv2.imread('../data/box.png')          # queryImage  
    img2 = cv2.imread('../data/box_in_scene.png') # trainImage
    model_fitting_type='homography' 
    draw_horizontal_layout = True 
#
if test_type == 'graf': 
    img1 = cv2.imread('../data/graf/img1.ppm') # queryImage
    img2 = cv2.imread('../data/graf/img3.ppm') # trainImage   img2, img3, img4
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    model_fitting_type='homography' 
    draw_horizontal_layout = True 
#
if test_type == 'kitti': 
    img1 = cv2.imread('../data/kitti06-12-color.png')
    img2 = cv2.imread('../data/kitti06-12-R-color.png')     
    #img2 = cv2.imread('../data/kitti06-13-color.png')     
    model_fitting_type='fundamental' 
    draw_horizontal_layout = False     
# 
if test_type == 'churchill': 
    img1 = cv2.imread('../data/churchill/1.ppm') 
    img2 = cv2.imread('../data/churchill/6.ppm')
    model_fitting_type='homography' 
    draw_horizontal_layout = True     
#
if test_type == 'mars': 
    # Very hard. This works with ROOT_SIFT, SUPERPOINT, CONTEXTDESC, LFNET, KEYNET ...     
    img1 = cv2.imread('../data/mars1.png') # queryImage
    img2 = cv2.imread('../data/mars2.png') # trainImage
    model_fitting_type='homography' 
    draw_horizontal_layout = True         
# 
if test_type == 'rat':   # add your test here 
    img1 = cv2.imread('test/data/graf1.png') 
    img2 = cv2.imread('test/data/graf2.png')
    model_fitting_type='homography' 
    draw_horizontal_layout = True     
    
if img1 is None:
    raise IOError('Cannot find img1')    
if img2 is None: 
    raise IOError('Cannot find img2')  
    
#============================================
# Transform Images (Optional)
#============================================  
    
M  = None     # rotation matrix on first image, if used 
H  = None     # homography matrix on first image, if used 
M2 = None     # rotation matrix on second image, if used 
H2 = None     # homography matrix on second image, if used 
    
# optionally apply a transformation to the first image 
if False: 
    img1, img1_box, M = rotate_img(img1, angle=20, scale=1.0)  # rotation and scale       
    #img1, img1_box, H = transform_img(img1, rotx=0, roty=-40, rotz=0, tx=0, ty=0, scale=1, adjust_frame=True) # homography 
    
    
# optionally regenerate the second image (override) by transforming the first image with a rotation or homography (here you have a ground-truth)
# N.B.: this procedure does not generate additional 'outlier-background' features: matching is much easier without a 'disturbing' 'background'. 
#       In order to add/generate a disturbing background, you can use the function add_background() (reported below)
if False: 
    #img2, img2_box, M2 = rotate_img(img1, angle=0, scale=1.0)  # rotation and scale       
    img2, img2_box, H2 = transform_img(img1, rotx=20, roty=30, rotz=40, tx=0, ty=0, scale=1.05, adjust_frame=True)   # homography 
    # optionally add a random background in order to generate 'outlier' features
    img2 = add_background(img2, img2_box, img_background=None) 


#============================================
# Init Feature Tracker   
#============================================  

num_features=2000 

#tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

# Get the type from the user input
main_type = sys.argv[1]
# select your tracker configuration (see the file feature_tracker_configs.py) 
# if main_type == 'superpoint':
#     tracker_config = FeatureTrackerConfigs.SUPERPOINT
# elif main_type == 'disk':
#     tracker_config = FeatureTrackerConfigs.DISK
# elif main_type == 'keynet':
#     tracker_config = FeatureTrackerConfigs.KEYNET
# # elif main_type == 'r2d2':
# #     tracker_config = FeatureTrackerConfigs.KEYNET
# elif main_type == 'tomasi':
#     tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
# elif main_type == 'context':
#     tracker_config = FeatureTrackerConfigs.CONTEXTDESC
tracker_config = FeatureTrackerConfigs.SUPERPOINT
tracker_config['num_features'] = num_features
tracker_config['match_ratio_test'] = 0.8        # 0.7 is the default in feature_tracker_configs.py
tracker_config['tracker_type'] = tracker_type
print('feature_manager_config: ',tracker_config)

feature_tracker = feature_tracker_factory(**tracker_config)

#============================================
# Compute keypoints and descriptors  
#============================================  
    
# loop for measuring time performance 
N=1
for i in range(N):
    
    # Find the keypoints and descriptors in img1
    kps1, des1 = feature_tracker.detectAndCompute(img1)
    
    timer.start()
    # Find the keypoints and descriptors in img2    
    kps2, des2 = feature_tracker.detectAndCompute(img2)
    # Find matches    
    idx1, idx2 = feature_tracker.matcher.match(des1, des2)
    timer.refresh()


print('#kps1: ', len(kps1))
if des1 is not None: 
    print('des1 shape: ', des1.shape)
print('#kps2: ', len(kps2))
if des2 is not None: 
    print('des2 shape: ', des2.shape)    

print('number of matches: ', len(idx1))

# Convert from list of keypoints to an array of points 
kpts1 = np.array([x.pt for x in kps1], dtype=np.float32) 
kpts2 = np.array([x.pt for x in kps2], dtype=np.float32)

# Get keypoint size 
kps1_size = np.array([x.size for x in kps1], dtype=np.float32)  
kps2_size = np.array([x.size for x in kps2], dtype=np.float32) 

# Build arrays of matched keypoints, descriptors, sizes 
kps1_matched = kpts1[idx1]
des1_matched = des1[idx1][:]
kps1_size = kps1_size[idx1]

kps2_matched = kpts2[idx2]
des2_matched = des2[idx2][:]
kps2_size = kps2_size[idx2]

# compute sigma mad of descriptor distances
#sigma_mad, dists = descriptor_sigma_mad(des1_matched,des2_matched,descriptor_distances=feature_tracker.descriptor_distances)
#print('3 x sigma-MAD of descriptor distances (all): ', 3 * sigma_mad)


#============================================
# Model fitting for extrapolating inliers 
#============================================  

hom_reproj_threshold = 3.0  # threshold for homography reprojection error: maximum allowed reprojection error in pixels (to treat a point pair as an inlier)
fmat_err_thld = 3.0         # threshold for fundamental matrix estimation: maximum allowed distance from a point to an epipolar line in pixels (to treat a point pair as an inlier)  

# Init inliers mask
mask = None 
   
h1,w1 = img1.shape[:2]  
if kps1_matched.shape[0] > 10:
    print('model fitting for',model_fitting_type)
    if model_fitting_type == 'homography': 
        # If enough matches are found, they are passed to find the perpective transformation. Once we get the 3x3 transformation matrix, 
        # we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it on img2.  
        # N.B.: this can be properly applied only when the view change corresponds to a proper homography transformation between the two sets of keypoints 
        #       e.g.: keypoints lie on a plane, view change corresponds to a pure camera rotation  
        H, mask = cv2.findHomography(kps1_matched, kps2_matched, cv2.RANSAC, ransacReprojThreshold=hom_reproj_threshold)   
        if img1_box is None: 
            img1_box = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
        else:
            img1_box = img1_box.reshape(-1,1,2)     
        pts_dst = cv2.perspectiveTransform(img1_box,H)
        # draw the transformed box on img2  
        img2 = cv2.polylines(img2,[np.int32(pts_dst)],True,(0, 0, 255),3,cv2.LINE_AA)    
        
        if mask is not None: 
            # ravel() return a contiguous flattened array.
            # A 1-D array, containing the elements of the input, is returned. 
            mask_idxs = (mask.ravel() == 1) 

            # Only use the inliers
            kps1_inliers = kps1_matched[mask_idxs]
            kps2_inliers = kps2_matched[mask_idxs]
        print(f'Originally there are {len(kps1)} matches. There are {len(kps1_inliers)} inliers after the RANSAC.')
        compute_hom_reprojection_error(H, kps1_inliers, kps2_inliers, mask)
        #print('reprojection error: ', reprojection_error)
    else:  
        F, mask = cv2.findFundamentalMat(kps1_matched, kps2_matched, cv2.RANSAC, fmat_err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
else:
    mask = None 
    print('Not enough matches are found for', model_fitting_type)
    
    
#============================================
# Drawing  
#============================================  

img_matched_inliers = None 
if mask is not None:    
    # Build arrays of matched inliers 
    mask_idxs = (mask.ravel() == 1)    
    
    kps1_matched_inliers = kps1_matched[mask_idxs]
    kps1_size_inliers = kps1_size[mask_idxs]
    des1_matched_inliers  = des1_matched[mask_idxs][:]    
    kps2_matched_inliers = kps2_matched[mask_idxs]   
    kps2_size_inliers = kps2_size[mask_idxs]    
    des2_matched_inliers  = des2_matched[mask_idxs][:]        
    print('num inliers: ', len(kps1_matched_inliers))
    print('inliers percentage: ', len(kps1_matched_inliers)/max(len(kps1_matched),1.)*100,'%')
        
    #sigma_mad_inliers, dists = descriptor_sigma_mad(des1_matched_inliers,des2_matched_inliers,descriptor_distances=feature_tracker.descriptor_distances)
    #print('3 x sigma-MAD of descriptor distances (inliers): ', 3 * sigma_mad)  
    #print('distances: ', dists)  
    img_matched_inliers = draw_feature_matches_horizontally(img1, img2, kps1_inliers, kps2_inliers)
    fig2 = MPlotFigure(img_matched_inliers, title='Inlier matches')
    MPlotFigure.show()
    exit()
    img_matched_inliers = draw_feature_matches(img1, img2, kps1_matched_inliers, kps2_matched_inliers, kps1_size_inliers, kps2_size_inliers,draw_horizontal_layout)    
                          
                          
img_matched = draw_feature_matches(img1, img2, kps1_matched, kps2_matched, kps1_size, kps2_size,draw_horizontal_layout)
                          
                                                
#fig1 = MPlotFigure(img_matched, title='All matches', main_type=main_type)
fig1 = MPlotFigure(img_matched, title='All matches', main_type = main_type)
if img_matched_inliers is not None: 
    fig2 = MPlotFigure(img_matched_inliers, title='Inlier matches', main_type=main_type)
MPlotFigure.show()
