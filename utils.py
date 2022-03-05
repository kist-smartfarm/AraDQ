#!/usr/bin/python
# -*- coding: utf-8 -*- 

#    Unseok Lee - Utils for image pre-processing and etcs.
#    Copyright (C) 2018 Unseok Lee
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details. 
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from scipy.spatial import distance
import cv2
import numpy as np
import colorbalance


def check_card_position(actual_colors_std):
    if any(actual_colors_std>90):
        print ('card damaged')
        grid_size=[4,6]
        return grid_size, False
    else:
        print ('ok')
        return [6,4],True

def color_correction(card=None,image=None,output_filename=None):
    
    card_damaged = False
    card_rotated = False
    correction_error = 0
    
    CardRGB = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    ImageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #CardRGB=card
    #ImageRGB=image
    
    actual_colors, actual_colors_std = colorbalance.get_colorcard_colors(CardRGB,grid_size=[6, 4]) #grid_cols, grid_rows
    
    new_grid_size,isSkip=check_card_position(actual_colors_std)
    if not isSkip:
        actual_colors, actual_colors_std = colorbalance.get_colorcard_colors(CardRGB,grid_size=new_grid_size)
    
    #true_colors = colorbalance.ColorCheckerRGB_CameraTrax
    true_colors = colorbalance.ColorCheckerRGB_CameraTrax_modified
   
    iter = 0
    actual_colors2 = actual_colors
    Check = True
    while Check:
        print (iter)
        iter = iter + 1
        color_alpha, color_constant, color_gamma = colorbalance.get_color_correction_parameters(true_colors,actual_colors2,'gamma_correction')
        corrected_colors = colorbalance._gamma_correction_model(actual_colors2, color_alpha, color_constant, color_gamma)
        diff_colors = true_colors - corrected_colors
        errors = np.sqrt(np.sum(diff_colors * diff_colors, axis=0)).tolist()
        # Sometimes, although card detection is OK (Acc is high), optimization for
        # color corection fails (high error). In this case, actual_colors are changed
        # slightly an dcorrection is repeated 
        #if Acc > 0.4 and np.mean(errors) > 40 and iter < 6:
        if np.mean(errors) > 40 and iter < 6:
            
            actual_colors2 = actual_colors + np.random.rand(3,24)
            
            #print('Corrction error high, correcting again....!')
        else:
            Check = False
   
    correction_error = round((np.mean(errors)/255)*10000)/float(100)
    print ('final_error : ', correction_error)
    #if correction_error < 50:  # equivalent to 20% error
    if correction_error < 100 :
        ImageRGBCorrected = colorbalance.correct_color(ImageRGB, color_alpha,color_constant, color_gamma)
        # get back to RGB order for OpenCV
        ImageCorrected = cv2.cvtColor(ImageRGBCorrected, cv2.COLOR_RGB2BGR)
        #if not os.path.exists(os.path.dirname(output_filename)):
        #    os.makedirs(os.path.dirname(output_filename))
        cv2.imwrite(output_filename,ImageCorrected)
        #cv2.imwrite('output3.png',ImageCorrected)
        
        print ('Done')
    return correction_error, ImageCorrected

def correct_distortion(top_left=None, bottom_left=None, top_right=None, bottom_right=None,img=None, mode=None, actual_size_w=4.7, actual_size_h = 4.7): #actual size of refernece, unit - cm
    
    
    if mode=='colorcard':
        a1 = distance.euclidean(top_left, bottom_left)
        a2 = distance.euclidean(top_right, bottom_right)
        
        b1 = distance.euclidean(top_left, top_right)
        b2 = distance.euclidean(bottom_left, bottom_right)
        temp_img=img
        
        temp_img = cv2.circle(temp_img, top_left, 10, (0,0,0), -1)
        temp_img = cv2.circle(temp_img, top_right, 10, (0,0,255), -1)
        temp_img = cv2.circle(temp_img, bottom_left, 10, (0,0,255), -1)
        temp_img = cv2.circle(temp_img, bottom_right, 10, (0,0,255), -1)
        
        
        a=max(a1,a2)
        b=max(b1,b2)
        w = int(b)
        h = int(a)
    
        shape=''
        
        print ('width:',w,'height:',h)
        
        if w>=h:
            new_top_left=[0,0]
            new_bottom_left = [0,4.4]
            new_top_right= [6.7,0]
            new_bottom_right=[6.7,4.4]
    
        else:
            
            new_top_left=[0,0]
            new_bottom_left = [0,6.7]
            new_top_right= [4.4,0]
            new_bottom_right=[4.4,6.7]
    elif mode=='qr':
        
        temp_img=img
        temp_img = cv2.circle(temp_img, top_left, 10, (0,0,0), -1)
        temp_img = cv2.circle(temp_img, top_right, 10, (0,0,255), -1)
        temp_img = cv2.circle(temp_img, bottom_left, 10, (0,0,255), -1)
        temp_img = cv2.circle(temp_img, bottom_right, 10, (0,0,255), -1)
        
        new_top_left=[0,0]
        new_bottom_left = [0,actual_size_h]
        new_top_right= [actual_size_w,0]
        new_bottom_right=[actual_size_w,actual_size_h]



    pts1 = np.float32([list(top_left),list(bottom_left),list(top_right),list(bottom_right)])
    pts2 = np.float32([new_top_left,new_bottom_left,new_top_right,new_bottom_right])
    H, status = cv2.findHomography(pts1,pts2)

    return H,temp_img
