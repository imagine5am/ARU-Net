from __future__ import print_function, division

import cv2
import random
from page_xml.xmlPAGE import pageData
from shapely.geometry import LineString
from utils import polyapprox as pa

import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from scipy import misc
from skimage import io
from pix_lab.util.util import load_graph

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=0.33, mode='L'):
        self.graph = load_graph(path_to_pb)
        self.img_list = img_list
        self.scale = scale
        self.mode = mode

    def inference(self, print_result=True, save_result=False, gpu_device="0"):
        val_size = len(self.img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        
        retVal = []
        
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            print("Start Inference")
            timeSum = 0.0
            for step in range(0, val_size):
                aTime = time.time()
                aImgPath = self.img_list[step]
                print("Image: {:} ".format(aImgPath))
                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                print(
                    "Resolution: h {:}, w {:} ".format(batch_x.shape[1],batch_x.shape[2]))
                # Run validation
                aPred = sess.run(predictor,
                                       feed_dict={x: batch_x})
                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime
                print(
                    "Update time: {:.2f} ms".format(curTime))
                if print_result:
                    n_class = aPred.shape[3]
                    channels = batch_x.shape[3]
                    fig = plt.figure()
                    for aI in range(0, n_class+1):
                        if aI == 0:
                            a = fig.add_subplot(1, n_class+1, 1)
                            if channels == 1:
                                plt.imshow(batch_x[0, :, :, 0], cmap=plt.cm.gray)
                            else:
                                plt.imshow(batch_x[0, :, :, :])
                            a.set_title('input')
                        else:
                            a = fig.add_subplot(1, n_class+1, aI+1)
                            plt.imshow(aPred[0,:, :,aI-1], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
                            # misc.imsave('out' + str(aI) + '.jpg', aPred[0,:, :,aI-1])
                            a.set_title('Channel: ' + str(aI-1))
                
                img_name = os.path.basename(aImgPath)
                save_name = os.path.splitext(img_name)[0]
                
                if save_result:
                    n_class = aPred.shape[3]
                    # aPred[0,:, :,0]
                    '''
                    for aI in range(0, n_class):        
                        save_loc = os.path.join('out', str(aI)+'_'+save_name+'.jpg')
                        io.imsave(save_loc, aPred[0,:, :,aI-1])
                    '''
                    save_loc = os.path.join('out', str(1)+'_'+save_name+'.jpg')
                    io.imsave(save_loc, aPred[0,:, :,0])
                    # print('aPred[0,:, :,0].shape:', aPred[0,:, :,0].shape)
                    print('To go on just CLOSE the current plot.')
                    plt.show()
                    
                self.gen_page(in_img_path=aImgPath, line_mask=aPred[0,:, :,0], id=save_name)
                retVal.append(aPred)
                
            self.output_epoch_stats_val(timeSum/val_size)

            print("Inference Finished!")

            return retVal

    def output_epoch_stats_val(self, time_used):
        print(
            "Inference avg update time: {:.2f} ms".format(time_used))

    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        if scale != 1:
            sImg = misc.imresize(aImg, scale, interp='bicubic')
            fImg = sImg
        else:
            fImg = aImg
            
        if len(fImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)

        return fImg
    
    def gen_page(self, in_img_path, line_mask, id):
        in_img = cv2.imread(in_img_path)
        (in_img_rows, in_img_cols, _) = in_img.shape
        # print('line_mask.shape:', line_mask.shape)
        
        cScale = np.array(
            [in_img_cols / line_mask.shape[1], in_img_rows / line_mask.shape[0]]
        )
        
        page = pageData(os.path.join('out', id + ".xml"))
        page.new_page(os.path.basename(in_img_path), str(in_img_rows), str(in_img_cols))
        
        kernel = np.ones((5, 5), np.uint8)
        validValues = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        
        lines = line_mask.copy()
        lines[line_mask > 0.1] = 1
        lines = lines.astype(np.uint8)
        
        # plt.axis("off")
        # plt.imshow(lines, cmap='gray')
        # plt.show()
    
        r_id = 0
        lin_mask = np.zeros(line_mask.shape, dtype="uint8")
        
        reg_mask = np.ones(line_mask.shape, dtype="uint8")
        res_ = cv2.findContours(
            reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(res_) == 2:
            contours, hierarchy = res_
        else:
            _, contours, hierarchy = res_
            
        for cnt in contours:
            min_area = 0.01
            # --- remove small objects
            if cnt.shape[0] < 4:
                continue
            if cv2.contourArea(cnt) < min_area * line_mask.shape[0]:
                continue

            rect = cv2.minAreaRect(cnt)
            # --- soft a bit the region to prevent spikes
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # box = np.array((rect[0][0], rect[0][1], rect[1][0], rect[1][1])).astype(int)
            r_id = r_id + 1
            approx = (approx * cScale).astype("int32")
            reg_coords = ""
            for x in approx.reshape(-1, 2):
                reg_coords = reg_coords + " {},{}".format(x[0], x[1])
                
            cv2.fillConvexPoly(lin_mask, points=cnt, color=(1, 1, 1))
            lin_mask = cv2.erode(lin_mask, kernel, iterations=1)
            lin_mask = cv2.dilate(lin_mask, kernel, iterations=1)
            reg_lines = lines * lin_mask
        
            resl_ = cv2.findContours(
                reg_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(resl_) == 2:
                l_cont, l_hier = resl_
            else:
                _, l_cont, l_hier = resl_
            
            # IMPORTANT l_cont, l_hier
            if len(l_cont) == 0:
                continue
                
            # --- Add region to XML only is there is some line
            uuid = ''.join(random.choice(validValues) for _ in range(4))
            text_reg = page.add_element(
                'TextRegion', "r" + uuid + "_" +str(r_id), 'full_page', reg_coords.strip()
            )
            n_lines = 0
            for l_id, l_cnt in enumerate(l_cont):
                if l_cnt.shape[0] < 4:
                    continue
                if cv2.contourArea(l_cnt) < 0.01 * line_mask.shape[0]:
                    continue
                # --- convert to convexHull if poly is not convex
                if not cv2.isContourConvex(l_cnt):
                    l_cnt = cv2.convexHull(l_cnt)
                lin_coords = ""
                l_cnt = (l_cnt * cScale).astype("int32")
                # IMPORTANT
                (is_line, approx_lin) = self._get_baseline(in_img, l_cnt)
                
                if is_line == False:
                    continue
                
                is_line, l_cnt = build_baseline_offset(approx_lin, offset=50)
                if is_line == False:
                    continue
                for l_x in l_cnt.reshape(-1, 2):
                    lin_coords = lin_coords + " {},{}".format(
                        l_x[0], l_x[1]
                    )
                uuid = ''.join(random.choice(validValues) for _ in range(4))
                text_line = page.add_element(
                    "TextLine",
                    "l" + uuid + "_" + str(l_id),
                    'full_page',
                    lin_coords.strip(),
                    parent=text_reg,
                )
                # IMPORTANT
                baseline = pa.points_to_str(approx_lin)
                page.add_baseline(baseline, text_line)
                n_lines += 1
        page.save_xml()
        
        # plt.axis("off")
        # plt.imshow(lines, cmap='gray')
        # plt.show()
        
    def _get_baseline(self, Oimg, Lpoly):
        """
        """
        # --- Oimg = image to find the line
        # --- Lpoly polygon where the line is expected to be
        minX = Lpoly[:, :, 0].min()
        maxX = Lpoly[:, :, 0].max()
        minY = Lpoly[:, :, 1].min()
        maxY = Lpoly[:, :, 1].max()
        mask = np.zeros(Oimg.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, Lpoly, (255, 255, 255))
        res = cv2.bitwise_and(Oimg, mask)
        bRes = Oimg[minY:maxY, minX:maxX]
        bMsk = mask[minY:maxY, minX:maxX]
        bRes = cv2.cvtColor(bRes, cv2.COLOR_RGB2GRAY)
        _, bImg = cv2.threshold(bRes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, cols = bImg.shape
        # --- remove black halo around the image
        bImg[bMsk[:, :, 0] == 0] = 255
        Cs = np.cumsum(abs(bImg - 255), axis=0)
        maxPoints = np.argmax(Cs, axis=0)
        Lmsk = np.zeros(bImg.shape)
        points = np.zeros((cols, 2), dtype="int")
        # --- gen a 2D list of points
        for i, j in enumerate(maxPoints):
            points[i, :] = [i, j]
        # --- remove points at post 0, those are very probable to be blank columns
        points2D = points[points[:, 1] > 0]
        if points2D.size <= 15:
            # --- there is no real line
            return (False, [[0, 0]])
        
        # --- take only 100 points to build the baseline
        max_vertex = 30
        num_segments = 4
        if points2D.shape[0] > max_vertex:
            points2D = points2D[
                np.linspace(
                    0, points2D.shape[0] - 1, max_vertex, dtype=np.int
                )
            ]
        (approxError, approxLin) = pa.poly_approx(
            points2D, num_segments, pa.one_axis_delta
        )
        
        approxLin[:, 0] = approxLin[:, 0] + minX
        approxLin[:, 1] = approxLin[:, 1] + minY
        return (True, approxLin)
    
def build_baseline_offset(baseline, offset=50):
    """
    build a simple polygon of width $offset around the
    provided baseline, 75% over the baseline and 25% below.
    """
    try:
        line = LineString(baseline)
        up_offset = line.parallel_offset(offset * 0.75, "right", join_style=2)
        bot_offset = line.parallel_offset(offset * 0.25, "left", join_style=2)
    except:
        #--- TODO: check if this baselines can be saved
        return False, None
    if (
        up_offset.type != "LineString"
        or up_offset.is_empty == True
        or bot_offset.type != "LineString"
        or bot_offset.is_empty == True
    ):
        return False, None
    else:
        up_offset = np.array(up_offset.coords).astype(np.int)
        bot_offset = np.array(bot_offset.coords).astype(np.int)
        return True, np.vstack((up_offset, bot_offset))
