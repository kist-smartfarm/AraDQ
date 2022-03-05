#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, jsonify,request
import os
import cv2
import numpy as np
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras.models import Model,model_from_json
import tensorflow as tf
import glob
import csv
from utils import color_correction,correct_distortion
from scipy.spatial import distance
from segmentation import predict
from scipy.spatial import ConvexHull
import datetime
import matplotlib
matplotlib.use('TkAgg') # In the case of Mac OS X (for python 2.x)
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

app = Flask(__name__)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

@app.route('/analyze',methods=["POST"])
def analyze():
    images_path=[]
    upload_files = request.files.getlist('files')
    cur_time = datetime.datetime.now()
    result_dir=cur_time.strftime('%Y_%m_%d_%H_%M')
    
    save_path = os.path.join(os.getcwd(),'results',result_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    for file in upload_files:
        fname = file.filename
        ext = fname.split('.')[-1]
        
        file.save(os.path.join(save_path, fname))
        final_path = os.path.join(save_path, fname)
        images_path.append(final_path)
    print (images_path)
    mode = 'colorcard'
    w_size_qr = 5 #cm
    h_size_qr = 6 #cm

    lights=[]
    darks=[]
    greens=[]
    gccs=[]
    convexhull_areas=[]
    projected_areas=[]
    contour_peri_open=[]
    contour_peri_close=[]
    convexhull_perimeters=[]
    detail_filename=[]
    # GRVIs =[]
    # VARIs = []
    ExGIs = []




    print ("Loading...")
    #detection of color card
    weight_path = os.path.join(os.getcwd(),"weights", "colorcard",'color_card_model.h5')
    color_card_model = models.load_model(weight_path, backbone_name='resnet152')
    print ('color card model is loaded.')
    #segmentation 
    checkpoint_dir = os.path.join(os.getcwd(),"weights",'ara_segmetnation')
    with open(os.path.join(checkpoint_dir, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
    segmentation_model = model_from_json(loaded_model_json)
    weight_list = sorted(glob.glob(os.path.join(checkpoint_dir, "*.h5")))
    segmentation_model.load_weights(weight_list[-1])
    print ('segmentation model is loaded.')

   

    for img_path in images_path:
        all_fname= os.path.split(img_path)
        all_fname= all_fname[-1]
        
        fname = all_fname[:-4]
        fname_ext = all_fname[-3:]
        ret_img, ret_mask=predict.custom_predict(model=segmentation_model,image_path=img_path,img_size=1024.0)
        cv2.imwrite(os.path.join(save_path,all_fname),ret_img)
        cv2.imwrite(os.path.join(save_path,fname + '_mask.' + fname_ext),ret_mask)
        img = read_image_bgr(os.path.join(save_path,all_fname))

        if mode=='colorcard':
            for i in range(0,4):
                img=np.rot90(img,i)        
                gray= cv2.imread(os.path.join(save_path,fname + '_mask.' + fname_ext),0)
                gray=np.rot90(gray,i)
                draw = img.copy()
                draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                image = preprocess_image(img)
                image, scale = resize_image(img)

                boxes, scores, labels = color_card_model.predict_on_batch(np.expand_dims(image, axis=0))
                print ('colorcard is predicted.')
                predicted_labels = labels[0]
                scores=scores[0]
                boxes /= scale
                idx=np.argsort(-scores) 
                new_idx=[]
                temp_brown=[]
                temp_white=[]
                temp_cyan=[]
                temp_black=[]
                for i in idx:
                    if predicted_labels[i]==0: # brown
                        temp_brown.append(i)
                    elif predicted_labels[i]==1: # white
                        temp_white.append(i)
                    elif predicted_labels[i]==2: # cyan
                        temp_cyan.append(i)
                    elif predicted_labels[i]==3: # black
                        temp_black.append(i)
                # Error check
                if len(temp_brown)==0:
                    print ('colorcard_brown error')
                    continue
                else:
                    new_idx.append(temp_brown[0])

                if len(temp_white)==0:
                    print ('colorcard_white error',all_fname)
                    continue
                else:
                    new_idx.append(temp_white[0])

                if len(temp_cyan)==0:
                    print ('colorcard_cyan error',all_fname)
                    continue
                else:
                    new_idx.append(temp_cyan[0])

                if len(temp_black)==0:
                    print ('colorcard_black error',all_fname)
                    continue

                else:
                    new_idx.append(temp_black[0])
            
                if len(new_idx)==4:
                    print ('COLORCARD OK')
                    break
            draw2=img.copy()    
            box_centers=[]    
            for i in new_idx:
                b=boxes[0,i].astype(int) 
                center_x = int((b[0] + b[2]) / 2)
                center_y = int((b[1] + b[3]) / 2)

                box_centers.append( (center_x, center_y) )
            
            if len(box_centers)==4:
                dists=[]

                for bb in box_centers:
                    dist = distance.euclidean((0,0), bb) 
                    dists.append(dist)
                
                if np.argmin(dists)==0:
                    top_left = box_centers[0]
                    bottom_left = box_centers[1]
                    top_right = box_centers[2]
                    bottom_right = box_centers[3]

                elif np.argmin(dists)==1: 
                    top_left = box_centers[1]
                    bottom_left = box_centers[3]
                    top_right = box_centers[0]
                    bottom_right = box_centers[2]
                elif np.argmin(dists)==2:
                    top_left = box_centers[2]
                    bottom_left = box_centers[0]
                    top_right = box_centers[3]
                    bottom_right = box_centers[1]
                elif np.argmin(dists)==3:
                    top_left = box_centers[3]
                    bottom_left = box_centers[2]
                    top_right = box_centers[1]
                    bottom_right = box_centers[0]

        elif mode=='qr':
            img_qr = cv2.imread(img_path,1)
            gray= cv2.imread(os.path.join(save_path,fname + '_mask.' + fname_ext),0)
               
            draw = img_qr
            draw2=img_qr.copy()    
            data=decode(img_qr)
            print ('QR code length:',len(data))
            if len(data)==0: 
                print ('qr_error')
                continue
    
            points=[]
            dists=[]
            p1 = ( int( data[0].polygon[0].x / scale ), int(data[0].polygon[0].y / scale))
            p2 = (  int(data[0].polygon[1].x /  scale), int(data[0].polygon[1].y / scale))
            p3 = ( int(data[0].polygon[2].x /  scale) , int(data[0].polygon[2].y / scale ) )
            p4 = ( int(data[0].polygon[3].x /  scale), int(data[0].polygon[3].y / scale) )

            points.append(p1)
            points.append(p2)
            points.append(p3)
            points.append(p4)

            for p in points:
                dist = distance.euclidean((0,0), p)
                dists.append(dist)
    
            if np.argmin(dists)==0:
                top_left = p1
                bottom_left = p2
                bottom_right = p3
                top_right = p4
            elif np.argmin(dists)==1: # top_left => white
                top_left = p2
                bottom_left = p3
                bottom_right = p4
                top_right = p1
            elif np.argmin(dists)==2:
                top_left = p3
                bottom_left = p4
                bottom_right = p1
                top_right = p2

            elif np.argmin(dists)==3:

                top_left = p4
                bottom_left = p1
                bottom_right = p2
                top_right = p3
        
        print ('top_left',top_left)

        H,draw=correct_distortion(top_left=top_left, bottom_left=bottom_left, top_right=top_right, bottom_right=bottom_right, img=draw,mode=mode)
        draw2=cv2.cvtColor(draw2,cv2.COLOR_BGR2RGB)
        inputImageCorners = np.float32([[[0,0],[0,draw2.shape[0]],[draw2.shape[1],0],[draw2.shape[1],draw2.shape[0]]]])
        outputCorners = cv2.perspectiveTransform(inputImageCorners,H)
        x,y,w,h = cv2.boundingRect(outputCorners)
        x2,y2,w2,h2 = cv2.boundingRect(inputImageCorners)

        origin=[x,y]
        scale_x = int(w2 / w)
        scale_y = int(h2 / h)
        scale=max(scale_x,scale_y)


        scale=max(scale_x,scale_y)

        t_x=-origin[0] * scale
        t_y=-origin[1] * scale
        S = np.float32([[scale,0,t_x],[0,scale,t_y],[0,0,1]])
        new_H= np.matmul(S,H)
        dst = cv2.warpPerspective(draw2, new_H, (w * scale,h*scale))
        print ('w',w)
        print ('w',h)
        print ('draw2',draw2.shape)
        print ('dst',dst.shape)

        
        dst_gray = cv2.warpPerspective(gray, new_H, (w * scale,h*scale))
        if mode =='colorcard':
            input_coor = np.float32([[[top_left[0],top_left[1]],[bottom_left[0],bottom_left[1]],[top_right[0],top_right[1]],[bottom_right[0],bottom_right[1]]]])
            output_coor = cv2.perspectiveTransform(input_coor,new_H)
            final_top_left= (int(output_coor[0][0][0]),int(output_coor[0][0][1]))
            final_bottom_left= (int(output_coor[0][1][0]),int(output_coor[0][1][1]))
            final_top_right=(int(output_coor[0][2][0]),int(output_coor[0][2][1]))
            final_bottom_right=(int(output_coor[0][3][0]),int(output_coor[0][3][1]))
            color_card= dst[final_top_left[1]:final_bottom_right[1],final_top_left[0]:final_bottom_right[0],:] 
            color_card=cv2.cvtColor(color_card,cv2.COLOR_RGB2BGR)
        dst=cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
        correction_path = os.path.join(save_path,'distortion_correction')
        if not os.path.exists(correction_path):
            os.makedirs(correction_path)

        cv2.imwrite(os.path.join(correction_path,all_fname),dst)
        cv2.imwrite(os.path.join(correction_path,fname + '_mask.' + fname_ext),dst_gray)
        corrected_img=cv2.imread(os.path.join(correction_path,all_fname),1)
        corrected_gray=cv2.imread(os.path.join(correction_path,fname + '_mask.' + fname_ext),0)
        print ('dis_corrected')
        final_color_crrection_error = ''
        if mode =='colorcard':
            temp_err,temp_ret_img=color_correction(card=color_card,image=corrected_img,output_filename='final_output.jpg')
           
            if temp_err <10 :
                temp_ret_img=temp_ret_img
                color_correction_path = os.path.join(save_path,'color_correction')
                if not os.path.exists(color_correction_path):
                    os.makedirs(color_correction_path)
                cv2.imwrite(os.path.join(color_correction_path,all_fname),temp_ret_img)
                final_color_crrection_error = temp_err
                print ('color_corrected')
            else:
                temp_color_card=[]
                for ii in range(1,4):
                    color_card=np.rot90(color_card,ii)
                    err, ret_img=color_correction(card=color_card,image=corrected_img,output_filename='final_output.jpg')
                    if err < 10:
                        temp_ret_img=ret_img
                        color_correction_path = os.path.join(save_path,'color_correction')
                        if not os.path.exists(color_correction_path):
                            os.makedirs(color_correction_path)
                        cv2.imwrite(os.path.join(color_correction_path,all_fname),temp_ret_img)
                        final_color_crrection_error = err
                        print ('color_corrected')
                        break
                    
                    if temp_err > err:
                        temp_err=err
                        temp_ret_img=ret_img
                        temp_color_card=color_card
                        color_correction_path = os.path.join(save_path,'color_correction')
                        if not os.path.exists(color_correction_path):
                            os.makedirs(color_correction_path)
                        cv2.imwrite(os.path.join(color_correction_path,all_fname),temp_ret_img)
                        final_color_crrection_error = temp_err
                        print ('color_corrected')

            print ('color error', temp_err)
            if final_color_crrection_error >=20:
                print ('colorcard error file_name:', all_fname)
                continue
            
            color_corrected_img = cv2.imread(os.path.join(color_correction_path,all_fname),1)
            corrected_img = color_corrected_img
            
        _, corrected_gray = cv2.threshold(corrected_gray,127,255, cv2.THRESH_BINARY)
        corrected_gray[np.where(corrected_gray==255)]=1
        label_img = label(corrected_gray,connectivity=2)
        regions = regionprops(label_img)
        dists2=[]
        for r in regions:
            d=distance.euclidean((0,0), (int(r.centroid[0]),int(r.centroid[1])))
            dists2.append(d)

        min_left= np.argsort(dists2)
        cnt=1
        for iii in min_left:
            convex_img = regions[iii].convex_image
            y1 = regions[iii].bbox[0]
            y2 =  regions[iii].bbox[2]
            x1 = regions[iii].bbox[1]
            x2 = regions[iii].bbox[3]
            dst4 = corrected_gray[y1:y2,x1:x2]
            dst3 = corrected_img[y1:y2,x1:x2]
            crop_result_path = os.path.join(save_path,'crop')
            if not os.path.exists(crop_result_path):
                os.makedirs(crop_result_path)
            filterd = dst4 * convex_img
            new_label_img = label(filterd,connectivity=2)
            new_regions = regionprops(new_label_img)
            regions_areas = [new_regions[k].area for k in range(0, np.max(new_label_img))]
            unique, counts = np.unique(new_label_img, return_counts=True)
            new_label_img[np.where(new_label_img!=1)] = 0
            new_label_img[np.where(new_label_img==1)] = 255
    
            cv2.imwrite(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '.' + fname_ext),dst3)
            cv2.imwrite(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_mask.' + fname_ext),new_label_img)
            
            dst3= cv2.imread(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '.' + fname_ext),1)
            dst4= cv2.imread(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_mask.' + fname_ext),0)

            for i in range(0,1):
                dst4=cv2.medianBlur(dst4,5)
            height, width = dst4.shape
            cv2.imwrite(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_mask.' + fname_ext),dst4)
            dst4= cv2.imread(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_mask.' + fname_ext),0)
            dst4[np.where(dst4<200)]=0
            hist_mask = cv2.calcHist([dst3],[1],dst4,[256],[0,256])
        
            new_cnt=0
            dark=0
            green=0
            light=0
            for i in hist_mask:
                new_cnt+=1

                if new_cnt <= 100:
                    dark=dark+i[0]
                elif new_cnt > 100 and new_cnt <= 128:
                    green=green +i[0]
                elif new_cnt > 128:
                    light=light + i[0]
            
            a=np.array([dark])
            b=np.array([green])
            c=np.array([light])
            
            masked_img = cv2.bitwise_and(dst3,dst3,mask=dst4)
            cv2.imwrite(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_masked_image.' + fname_ext),masked_img)
            b, g, r = cv2.split(masked_img)
        
            temp_b=[]
            for s in b:
                for sss in s:
                    if sss!=0:
                        temp_b.append(sss)
            temp_g=[]
            for s in g:
                for sss in s:
                    if sss!=0:
                        temp_g.append(sss)

            temp_r=[]
            for s in r:
                for sss in s:
                    if sss!=0:
                        temp_r.append(sss)
                        
            img_gcc= np.mean(temp_g) / (np.mean(temp_g) + np.mean(temp_b) + np.mean(temp_r))
            img_ExGI = 2 * np.mean(temp_g) - ( np.mean(temp_r) + np.mean(temp_b) )
            # img_GRVI = ( np.mean(temp_g) - np.mean(temp_r) ) / ( np.mean(temp_g) + np.mean(temp_r) )
            # img_VARI = ( np.mean(temp_g) - np.mean(temp_r) ) / ( np.mean(temp_g) + np.mean(temp_r) - np.mean(temp_b) )
            gccs.append(img_gcc)
            ExGIs.append(img_ExGI)
            _,contours, _ = cv2.findContours(dst4,2,1)
        
            pts=[]
            contour_perimeter_colosed=0
            contour_perimeter_open=0
        
            for contour in contours:

                for cont in contour:
                    pts.append(list(cont[0]))
                temp_contour_perimeter_closed= cv2.arcLength(contour,True)
                temp_contour_perimeter_open= cv2.arcLength(contour,False)
                
                #print contour_perimeter2
                contour_perimeter_colosed+=temp_contour_perimeter_closed
                contour_perimeter_open+=temp_contour_perimeter_open
                
    
            ptpts=np.array(pts)
            plt.figure(figsize=(10,10))
            plt.plot(ptpts[:,0],ptpts[:,1],'o')
        
           
            hull=ConvexHull(ptpts)
            #흰 배경 + 파란색 점 + 검정 convexhull
            hull_pts =[]
            for simplex in hull.simplices:
                plt.plot(ptpts[simplex,0],ptpts[simplex,1],'k-')
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(crop_result_path,fname + '_' + str(cnt)+ '_hull.png'),dpi=100)  
            
            print('hull')
            
            hull_area=hull.volume
            hull_perimeter = hull.area
            convexhull_areas.append(hull_area / (scale * scale))
            convexhull_perimeters.append(hull_perimeter / (scale) )
            contour_peri_close.append(contour_perimeter_colosed / (scale ) )
            contour_peri_open.append(contour_perimeter_open / (scale))
            darks.append(dark / (scale * scale))
            greens.append(green / (scale * scale))
            lights.append(light / (scale * scale))
            projected_areas.append( (dark+green+light) / (scale * scale) )
            detail_filename.append(fname + '_' + str(cnt))                                              
        
                
            cnt+=1
        # save as csv
        f = open(os.path.join(save_path,'final_result.csv'), 'w')
        w = csv.writer(f)
        w.writerow(['file_name', "dark (cm2)", 'green (cm2)','light (cm2)','gcc','ExGIs','convexhull_area (cm2)','projected_areas (cm2)','contour_peri_open(cm)','convexhull_perimeters(cm)'])


        for i in range(0,len(gccs)):
            w.writerow([detail_filename[i], darks[i], greens[i],lights[i],gccs[i],ExGIs[i],convexhull_areas[i],projected_areas[i],contour_peri_open[i],convexhull_perimeters[i]])
        f.close()


    return jsonify(flag='ok', msg=u'Finished.',path=save_path)









@app.route('/',methods=["GET"])
def main():
    return render_template('aradq_main.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 20001))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
