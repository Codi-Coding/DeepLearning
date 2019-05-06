# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Test for Stage 1: from product image + body segment +
    pose + face/hair predict a coarse result and product segment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc
import pickle as pkl
#import h5py
#import hdf5storage
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import matlab.engine
from utils import *
from utils2 import create_model
from composition_part import _load_image
from composition_part import process_raw_image


from composition_lower_part import _process_ratio
from composition_lower_part import process_raw_mask
from tps_transformer import tps_stn
from PIL import Image

import threading
import socket
import queue
import requests
import shutil

mat_engine = matlab.engine.start_matlab()

wait_count = 0
wait_queue = queue.Queue()


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


FLAGS = tf.app.flags.FLAGS
  
tf.flags.DEFINE_string("userId", "", "user id")
tf.flags.DEFINE_string("imageId", "", "user's image id")
tf.flags.DEFINE_string("upperId", "", "composed upper id")
tf.flags.DEFINE_string("lowerId", "", "composed lower id")
tf.flags.DEFINE_integer("isUpper", 0, "0 : no composed upper, 1 : composed upper")
tf.flags.DEFINE_integer("category", 0, "")
tf.flags.DEFINE_integer("switch", 1, "")
tf.flags.DEFINE_string("category_name", "", "category's name to change category")

tf.flags.DEFINE_string("result_dir_stage1", "testdata/" + FLAGS.userId + "/stage/",
                       "Folder containing the results of testing1.")
tf.flags.DEFINE_string("result_dir_stage2", "testdata/" + FLAGS.userId + "/output/composed_upper_images/",
                       "Folder containing the results of testing2.")
tf.flags.DEFINE_string("output", "testdata/" + FLAGS.userId + "/output", "")


tf.flags.DEFINE_string("checkpoint1", "model/" + "" + "/stage1/model-15000",
                       "Multi-task Encoder decoder generator")
tf.flags.DEFINE_string("checkpoint2", "model/" + "" + "/stage2/model-6000",
                       "refinement network")


tf.flags.DEFINE_string("input_dir", "testdata/" + FLAGS.userId + "/input",
                       "user image(upper) pickle, resizing tensor")

tf.flags.DEFINE_string("mall_name", "test", "shopping mall name")
tf.flags.DEFINE_string("prod_dir", "data/" + FLAGS.mall_name + "/" + FLAGS.category_name + "/",
                       "product image(tshirts) pickle, resizing tensor")
#tf.flags.DEFINE_string("text_of_interval", "../testdata/" + FLAGS.userId + "/interval_upper_data.txt",
#                       "interval of width")

tf.flags.DEFINE_integer("begin", "0", "")
tf.flags.DEFINE_integer("end", "1", "")
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS.mode = "test"


class model_stage1(object):
  def __init__(self, category_num, graph1):
    tf.reset_default_graph()
    self.graph = graph1
    with self.graph.as_default() as graph:
      batch_size = 1
      self.image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
      self.prod_image_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.body_segment_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 1])
      self.prod_segment_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 1])
      self.skin_segment_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.pose_map_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 18])
      self.model = create_model(self.prod_image_holder, self.body_segment_holder,
                      self.skin_segment_holder, self.pose_map_holder,
                      self.prod_segment_holder, self.image_holder, category_num)
      self.images = np.zeros((batch_size, 256, 192, 3))
      self.prod_images = np.zeros((batch_size, 256, 192, 3))
      self.body_segments = np.zeros((batch_size, 256, 192, 1))
      self.prod_segments = np.zeros((batch_size, 256, 192, 1))
      self.skin_segments = np.zeros((batch_size, 256, 192, 3))
      self.pose_raws = np.zeros((batch_size, 256, 192, 18))
      self.saver = tf.train.Saver()
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.session = tf.Session(config=config,graph=self.graph)
      print(category_num)
      if category_num == '1001':
        category_name = "men_tshirts"
        FLAGS.checkpoint1 = "model/" + category_name + "/stage1/model-15000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint1)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint1
        
        print("loading stage1 model from men_tshirts checkpoint")
      elif category_num == '1002':
        category_name = "men_nambang"
        FLAGS.checkpoint1 = "model/" + category_name + "/stage1/model-15000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint1)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint1
        print("loading stage1 model from men_nambang checkpoint")

      elif category_num == '1003':
        category_name = "men_long"
        FLAGS.checkpoint1 = "model/" + category_name + "/stage1/model-6000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint1)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint1
        print("loading stage1 model from men_long checkpoint")
      elif category_num == '1101':
        category_name = "men_pants"
        FLAGS.checkpoint1 = "model/" + category_name + "/stage1/model-15000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint1)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint1
        print("loading stage1 model from men_pants checkpoint")
      #if self.checkpoint == None:
      #  self.checkpoint = FLAGS.checkpoint1
      #print(FLAGS.checkpoint)
      print("22222")
      self.saver.restore(self.session, self.checkpoint)
      
    

  #def process_image_predict(self, param):
  #  result = self.session.run(param)
  #  return result
  def predict(self, param, dic):
    result = self.session.run(param, feed_dict=dic)
    return result
class model_stage2(object):
  def process_one_image(self, image, resize_height, resize_width, if_zero_one=False):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if if_zero_one:
      return image
    image = tf.image.resize_images(image,
                                  size=[resize_height, resize_width],
                                  method=tf.image.ResizeMethod.BILINEAR)
    return (image - 0.5) * 2.0

  def _process_image_stage2(self, prod_image, image_name, product_image_name, sess,
                        resize_width=192, resize_height=256):
    image_id = image_name[:-4]
    #image = scipy.misc.imread(FLAGS.image_dir + image_name)
    #prod_image = scipy.misc.imread(FLAGS.prod_image_dir + product_image_name)
    # sorry for the hard coded file path.
    coarse_image = scipy.misc.imread(FLAGS.result_dir_stage1 + 
                                    image_name + "_" +
                                    product_image_name + ".png")
    mask_output = scipy.misc.imread(FLAGS.result_dir_stage1 + 
                                    image_name + "_" +
                                    product_image_name + "_mask.png")
    #image = process_one_image(image, resize_height, resize_width)
    #prod_image = process_one_image(prod_image, resize_height, resize_width)
    coarse_image = self.process_one_image(coarse_image, resize_height, resize_width)
    mask_output = self.process_one_image(mask_output, resize_height,
                                    resize_width, True)
    # TPS transform
    # Here we use control points to generate 
    # We tried to learn the control points, but the network refuses to converge.
    tps_control_points = sio.loadmat(FLAGS.result_dir_stage1 +
                                    image_name + "_" +
                                    product_image_name +
                                    "_tps.mat")
    v = tps_control_points["control_points"]
    nx = v.shape[1]
    ny = v.shape[2]
    v = np.reshape(v, -1)
    v = np.transpose(v.reshape([1,2,nx*ny]), [0,2,1]) * 2 -1
    p = tf.convert_to_tensor(v, dtype=tf.float32)
    img = tf.reshape(prod_image, [1,256,192,3])

    tps_image = tps_stn(img, nx, ny, p, [256,192,3])

    tps_mask = tf.cast(tf.less(tf.reduce_sum(tps_image, -1), 3*0.95), tf.float32)
    
    [coarse_image, tps_image, mask_output, tps_mask] = sess.run(
                [coarse_image, tps_image, mask_output, tps_mask])
    
    
    return coarse_image, tps_image, mask_output, tps_mask
  def create_refine_generator(self, stn_image_outputs, gen_image_outputs):
    generator_input = tf.concat([stn_image_outputs, gen_image_outputs],
                                axis=-1)
    downsampled = tf.image.resize_area(generator_input, (256, 192), align_corners=False)
    net = slim.conv2d(downsampled, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope="g_256_conv1")
    net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope='g_256_conv2')
    net = slim.conv2d(net, 64, [3, 3], rate=1, normalizer_fn=slim.layer_norm,
                        activation_fn=lrelu, scope='g_256_conv3')
    net = slim.conv2d(net, 1, [1, 1], rate=1,
                        activation_fn=None, scope='g_1024_final')
    net = tf.sigmoid(net)
    return net
  def __init__(self, category_num, graph2):
    self.graph = graph2
    tf.reset_default_graph()
    with self.graph.as_default() as graph:
      batch_size = 1
      print(category_num)
      self.image_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.prod_image_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.prod_mask_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 1])
      self.coarse_image_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.tps_image_holder = tf.placeholder(
          tf.float32, shape=[batch_size, 256, 192, 3])
      self.images = np.zeros((batch_size, 256, 192, 3))
      self.prod_images = np.zeros((batch_size, 256, 192, 3))
      self.coarse_images = np.zeros((batch_size, 256, 192, 3))
      self.tps_images = np.zeros((batch_size, 256, 192, 3))
      self.mask_outputs = np.zeros((batch_size, 256, 192, 1))
      
      
      with tf.variable_scope("refine_generator") as scope:
        self.select_mask = self.create_refine_generator(self.tps_image_holder, self.coarse_image_holder)
        self.select_mask = self.select_mask * self.prod_mask_holder
        self.model_image_outputs = (self.select_mask * self.tps_image_holder +
                              (1 - self.select_mask) * self.coarse_image_holder)

      self.saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()
                                    if var.name.startswith("refine_generator")])                      
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.session = tf.Session(config=config,graph=self.graph)
      self.config = tf.ConfigProto(device_count = {'GPU':1})
      print("loading model from checkpoint")

      if category_num == '1001':
        category_name = "men_tshirts"
        FLAGS.checkpoint2 = "model/" + category_name + "/stage2/model-6000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint2)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint2
        print("loading stage2 model from men_tshirts checkpoint")
      elif category_num == '1002':
        category_name = "men_nambang"
        FLAGS.checkpoint2 = "model/" + category_name + "/stage2/model-6000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint2)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint2
        print("loading stage2 model from men_nambang checkpoint")
      elif category_num == '1003':
        category_name = "men_long"
        FLAGS.checkpoint2 = "model/" + category_name + "/stage2/model-60000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint2)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint2
        print("loading stage2 model from men_pants checkpoint")
      #if self.checkpoint == None:
      #  self.checkpoint = FLAGS.checkpoint2
      elif category_num == '1101':
        category_name = "men_pants"
        FLAGS.checkpoint2 = "model/" + category_name + "/stage2/model-6000"
        self.checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint2)
        if self.checkpoint == None:
          self.checkpoint = FLAGS.checkpoint2
        print("loading stage2 model from men_pants checkpoint")
      #if self.checkpoint == None:
      #  self.checkpoint = FLAGS.checkpoint2
      
      self.saver.restore(self.session, self.checkpoint)

      
    
  def process_image_predict(self, param):
    result = self.session.run(param)
    return result
  def predict(self, param, dic):
    result = self.session.run(param, feed_dict=dic)
    return result
# preprocess images for testing
'''
  stage2 module
'''
def deprocess_image(image, mask01 = False):
  if not mask01:
    image = image / 2 + 0.5
  return image
def select_category(num):
  return {'1001' : "men_tshirts",
          '1002' : "men_nambang",
          '1003' : "men_long",
          '1101' : "men_pants"
         }.get(num, "No data")
def GetFlask():
	print("GETFLASK")
	s = socket.socket()
	host = socket.gethostname()
	port = 12222
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.bind((host, port))
	s.setblocking(1)
	s.listen(5)
	c = None
	a = 1
	
	while True:
		print("#")
		print(a)
		print(c)
		if c is None or a == 1:
			 # Halts
			print( '[Waiting for connection...]')
			c, addr = s.accept() #  (socket object, address info) return
			print( 'Got connection from', addr)
			a = 0
		else:
			 # Halts
			#time.sleep(3)
			print( '[Waiting for response...]')
			wait_str = (c.recv(1024)).decode('utf-8') #여기서 멈춘다
			print(wait_str)
			print(len(wait_str))
			
			wait_list = wait_str.split()
			if len(wait_list) < 2 :
				print("continue")
				c=None
				a=1
				continue
			

			global wait_count
			
			wait_queue.put(wait_list)
			wait_count = wait_count+1
						
			if wait_str=='0':
				print("shutdown")
				return 1
			c = None	
			a=1
			 #c.send(q.encode('utf-8'))
def save_product_mask(raw, mask, composition_name, tag=""):
  
  mask = process_raw_mask(mask)
  fg = cv2.bitwise_and(raw, raw, mask=mask)
  info_data = {}
  info_data['product_mask'] = fg
  with open(FLAGS.result_dir_stage2 + composition_name + '1_pkl.pkl', 'wb') as f4:
    pkl.dump(info_data, f4, pkl.HIGHEST_PROTOCOL)
  if tag=='save':
    return fg
def save_model_mask(model_image, raw, mask, composition_name, interval, isUpper, tag=""):
  
  mask = process_raw_mask(mask)
  mask_reverse = cv2.bitwise_not(mask)
  print("IS upper " + str(isUpper))
  if isUpper == 0:
    print('Is upper = 0')
    bg = cv2.bitwise_and(model_image, model_image, mask=mask_reverse)
  elif isUpper == 1:
    roi = model_image[:256, interval:interval + 192]
    bg = cv2.bitwise_and(roi, roi, mask = mask_reverse)

  info_data = {}
  info_data['model_mask'] = bg
  with open(FLAGS.result_dir_stage2 + composition_name + '0_pkl.pkl', 'wb') as f5:
    pkl.dump(info_data, f5, pkl.HIGHEST_PROTOCOL)
  if tag == 'save':
    return bg  
def final_trim_image(final_image):
  image_height = final_image.shape[0]
  image_width = final_image.shape[1]
  check_width = image_height * 0.45
  cropping_interval = int((image_width - check_width) / 2)

  update_image = final_image[:, cropping_interval:-cropping_interval]

  return update_image
def final_process(model_image, composition_name, fg, bg, interval, isUpper, final):
    coarse_image = cv2.add(fg, bg)
    if isUpper == 0:
      result_dir = FLAGS.output + "/final_lower_images/"
      if final == 1:
        result_dir = FLAGS.output + "/final_images/"
        coarse_image = final_trim_image(coarse_image)
      #final_image = final_trim_image(coarse_image)
      scipy.misc.imsave(result_dir + composition_name + "final.png", coarse_image)
    elif isUpper == 1:
      model_image[:256, interval:interval + 192] = coarse_image
      result_dir = FLAGS.output + "/final_upper_images/"
      if final == 1:
        result_dir = FLAGS.output + "/final_images/"
        model_image = final_trim_image(model_image)
      #final_image = final_trim_image(model_image)
      scipy.misc.imsave(result_dir + composition_name + "final.png", model_image)
def inference(image_names, product_image_names, image_name, product_image_name, model1, model2, isUpper, i, j):
  stage1_start = time.time()
  
  if isUpper is 0:
    image_pkl_dir = FLAGS.input_dir + "/body_pickle/"
  elif isUpper is 1:
    image_pkl_dir = FLAGS.input_dir + "/upper_pickle/"

  FLAGS.prod_dir = "data/" + FLAGS.mall_name + "/" + FLAGS.category_name + "/"
  prod_image_pkl_dir = FLAGS.prod_dir + "pkl/"

  image_names.append(image_name)
  product_image_names.append(product_image_name)

  stage1_img_process_start = time.time()

  with open(image_pkl_dir + image_name + ".pkl", "rb") as f1:
    try:
      image_data = pkl.load(f1, encoding = "latin-1")
    except EOFError:
      pass
  with open(prod_image_pkl_dir + product_image_name + ".pkl", "rb") as f2:
    try:
      prod_image_data = pkl.load(f2, encoding = "latin-1")
    except EOFError:
      pass
  
  image = image_data['image']
  prod_image = prod_image_data['prod_image']
  pose_raw = image_data['pose_raw']
  body_segment = image_data['body_seg']
  prod_segment = image_data['prod_seg']
  skin_segment = image_data['skin_seg']
  
  stage1_img_process_end = time.time()
  print("[stage1] process image time : " + str(stage1_img_process_end - stage1_img_process_start))

  model1.images[j-i] = image
  model1.prod_images[j-i] = prod_image
  model1.body_segments[j-i] = body_segment
  model1.prod_segments[j-i] = prod_segment
  model1.skin_segments[j-i] = skin_segment
  model1.pose_raws[j-i] = pose_raw
  # inference
  feed_dict1 = {
      model1.image_holder: model1.images,
      model1.prod_image_holder: model1.prod_images,
      model1.body_segment_holder: model1.body_segments,
      model1.skin_segment_holder: model1.skin_segments,
      model1.prod_segment_holder: model1.prod_segments,
      model1.pose_map_holder: model1.pose_raws,
  }
  stage1_predict_start = time.time()
  [image_output, mask_output, loss, step] = model1.predict(
      [model1.model.image_outputs,
      model1.model.mask_outputs,
      model1.model.gen_loss_content_L1,
      model1.model.global_step],
      dic=feed_dict1)
  stage1_predict_end = time.time()
  print("[stage1] predict time : " + str(stage1_predict_end - stage1_predict_start))
  scipy.misc.imsave(FLAGS.result_dir_stage1 +
                    image_names[j] + "_" + product_image_names[j] + '.png',
                    (image_output[j] / 2.0 + 0.5))
  scipy.misc.imsave(FLAGS.result_dir_stage1 +
                    image_names[j] + "_" + product_image_names[j] + '_mask.png',
                    np.squeeze(mask_output[j]))
  
  sio.savemat(FLAGS.result_dir_stage1 +
              image_names[j] + "_" + product_image_names[j] + "_mask.mat",
              {"mask": np.squeeze(mask_output[j])})
  stage1_end = time.time()
  print("[stage1] process time : " + str(stage1_end - stage1_start))
  # matlab으로 경로정보를 보내기 위해서
  with open("connection.txt", "w") as f3:
    prod_image_dir = FLAGS.prod_dir + "images/"
    prod_image_file_name = ""
    if os.path.exists(prod_image_dir + product_image_names[j] + ".jpg"):
      prod_image_file_name = product_image_names[j] + ".jpg"
    elif os.path.exists(prod_image_dir + product_image_names[j] + ".png"):
      prod_image_file_name = product_image_names[j] + ".png"
      
    f3.write(image_names[j] +" " + prod_image_file_name + " " + prod_image_dir + " " + FLAGS.result_dir_stage1)

  mat_start = time.time()
  mat_engine.shape_context_warp(nargout=0)
  mat_end = time.time()
  print("[matlab] shape context warp time : " + str(mat_end - mat_start))

   # stage2
  stage2_start = time.time()
  stage2_image_process_start = time.time()
  
  process_image_graph = tf.Graph()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    (coarse_image, tps_image, mask_output, 
    tps_mask) = model2._process_image_stage2(prod_image, image_name, 
                                        product_image_name, sess)
  stage2_image_process_end = time.time()
  print("[stage2] process_image time : " + str(stage2_image_process_end - stage2_image_process_start))
  model2.images[j-i] = image
  model2.prod_images[j-i] = prod_image
  model2.coarse_images[j-i] = coarse_image
  model2.tps_images[j-i] = tps_image
  model2.mask_outputs[j-i] = np.expand_dims(mask_output, -1)

  #inference
  feed_dict2 = {
    model2.image_holder: model2.images,
    model2.prod_image_holder: model2.prod_images,
    model2.coarse_image_holder: model2.coarse_images,
    model2.tps_image_holder: model2.tps_images,
    model2.prod_mask_holder: model2.mask_outputs,
  }   
  stage2_predict_start = time.time()
  [image_output, sel_mask] = model2.predict([model2.model_image_outputs, model2.select_mask],
                            dic=feed_dict2)
  stage2_predict_end = time.time()
  print("[stage2] predict time : " + str(stage2_predict_end - stage2_predict_start))
  scipy.misc.imsave(FLAGS.result_dir_stage2 + image_names[j] +
                  "_" + product_image_names[j] + '_mask.png',
                  np.squeeze(model2.mask_outputs[j]))
  scipy.misc.imsave(FLAGS.result_dir_stage2 + image_names[j] +
                  "_" + product_image_names[j] + '_final.png',
                  (image_output[j]) / 2.0 + 0.5)
  stage2_end = time.time()
  print("[stage2] process time : " + str(stage2_end - stage2_start))
def main(unused_argv):
  batch_size = 1
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  # graph update
  graph_men_tshirts_stage1 = tf.Graph()
  graph_men_tshirts_stage2 = tf.Graph()
  graph_men_nambang_stage1 = tf.Graph()
  graph_men_nambang_stage2 = tf.Graph()
  graph_men_long_stage1 = tf.Graph()
  graph_men_long_stage2 = tf.Graph()

  graph_men_pants_stage1 = tf.Graph()
  graph_men_pants_stage2 = tf.Graph()

  # Loading model in memory
  men_tshirts_stage1 = model_stage1(category_num='1001', graph1=graph_men_tshirts_stage1)
  men_tshirts_stage2 = model_stage2(category_num='1001', graph2=graph_men_tshirts_stage2)
  men_nambang_stage1 = model_stage1(category_num='1002', graph1=graph_men_nambang_stage1)
  men_nambang_stage2 = model_stage2(category_num='1002', graph2=graph_men_nambang_stage2)
  men_long_stage1 = model_stage1(category_num='1003', graph1=graph_men_long_stage1)
  men_long_stage2 = model_stage2(category_num='1003', graph2=graph_men_long_stage2)
  men_pants_stage1 = model_stage1(category_num='1101', graph1=graph_men_pants_stage1)
  men_pants_stage2 = model_stage2(category_num='1101', graph2=graph_men_pants_stage2)
  #men_pants_stage1 = model_stage1(5)
  #men_pants_stage2 = model_stage2(5)
  #model
  model_dict = {'1001' : (men_tshirts_stage1, men_tshirts_stage2),
                '1002' : (men_nambang_stage1, men_nambang_stage2),
                '1003' : (men_long_stage1, men_long_stage2),
                '1101' : (men_pants_stage1, men_pants_stage2)
               }
  threading._start_new_thread(GetFlask,())

  # batch inference, can also be done one image per time.
  #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  
  while True:
    try:
      while wait_queue.qsize() != 0:
        
        
        queue_data = wait_queue.get()
        print(queue_data)

        #est_info = open(FLAGS.test_label).read().splitlines()
        #interval_list = open(FLAGS.text_of_interval).read().splitlines()
        image_data = {}
        prod_image_data = {}
        
        for i in range(FLAGS.begin, FLAGS.end, batch_size):
          image_names = []
          product_image_names = []

          for j in range(i, i + batch_size):
            #info = test_info[j].split()
            #print(info)
            FLAGS.userId = queue_data[0]
            FLAGS.imageId = queue_data[1]
            FLAGS.upperId = queue_data[2]
            FLAGS.lowerId = queue_data[3]
            FLAGS.isUpper = int(queue_data[4])
            FLAGS.category = queue_data[5]
            FLAGS.category_name = select_category(FLAGS.category)
            # make directory
            
            model1 = model_dict[FLAGS.category][0]
            model2 = model_dict[FLAGS.category][1]
            
            # path refresh
            image_name = queue_data[1] + "_0"
            try:
              os.mkdir("testdata/" + FLAGS.userId + "/stage")
              os.mkdir("testdata/" + FLAGS.userId + "/output")
            except:
              pass
            try:
              os.mkdir("testdata/" + FLAGS.userId + "/output/composed_images")
              os.mkdir("testdata/" + FLAGS.userId + "/output/final_upper_images")
              os.mkdir("testdata/" + FLAGS.userId + "/output/final_lower_images")
              os.mkdir("testdata/" + FLAGS.userId + "/output/final_images")
            except:
              pass
            FLAGS.result_dir_stage1 = "testdata/" + FLAGS.userId + "/stage/"
            FLAGS.result_dir_stage2 = "testdata/" + FLAGS.userId + "/output/composed_images/"

            FLAGS.input_dir = "testdata/" + FLAGS.userId + "/input"
            FLAGS.output = "testdata/" + FLAGS.userId + "/output"
            
            with open(FLAGS.input_dir + "/body_pickle/" + image_name + ".pkl", "rb") as f1:
              try:
                image_data = pkl.load(f1, encoding = "latin-1")
              except EOFError:
                pass


            #######후처리#######
            #1 앞부분 (상/하의 이미지 합성)  -> #2 뒷부분 final upper 와 lower 합성

            ######앞부분 (#1 상의 하의 따라 final_upper 또는 final_lower 생성)
            upper_name = FLAGS.upperId + "_1"
            lower_name = FLAGS.lowerId + "_1"
            upper_composition_name = image_name + "_" + upper_name + "_"
            lower_composition_name = image_name + "_" + lower_name + "_"
            full_composition_name = image_name + "_" + upper_name + "_" + lower_name + "_"

            middle_composition_name = ""
            if(FLAGS.isUpper == 0):
              print("Lower")
              middle_composition_name = lower_composition_name
              product_image_name = FLAGS.lowerId + '_1'
            elif(FLAGS.isUpper == 1):
              print("Upper")
              middle_composition_name = upper_composition_name
              product_image_name = FLAGS.upperId + '_1'
    

            model_image = _load_image(FLAGS.input_dir + "/body_resized/" + image_name + ".jpg")
            model_image_height = model_image.shape[0]
            model_image_width = model_image.shape[1]

            interval = int(image_data['resized_interval'])
            if not os.path.exists(FLAGS.result_dir_stage2 + middle_composition_name + "1_pkl.pkl"):
              #기존 합치고
            
              inference(image_names, product_image_names, image_name, 
                        product_image_name, model1, model2, FLAGS.isUpper, i, j)

              composition_raw = _load_image(FLAGS.result_dir_stage2 + middle_composition_name + "final.png")
              composition_mask = _load_image(FLAGS.result_dir_stage2 + middle_composition_name + "mask.png")

              ##하의 합성시 이미지 크기 확대해서
              if(FLAGS.isUpper == 0):
                composition_raw = _process_ratio(model_image_height, model_image_width,
                                  composition_raw)
                composition_mask = _process_ratio(model_image_height, model_image_width,
                                  composition_mask)


              save_model_mask(model_image, composition_raw, composition_mask, 
                            middle_composition_name, interval, FLAGS.isUpper, tag="")
              save_product_mask(composition_raw, composition_mask, middle_composition_name)
              #합성 수행
            
            post1 = time.time()
            with open(FLAGS.result_dir_stage2 + middle_composition_name + "0_pkl.pkl", "rb") as f1:
              try:
                model_mask = pkl.load(f1, encoding = "latin-1")
              except EOFError:
                print("pkl.load fail")
                break
            with open(FLAGS.result_dir_stage2 + middle_composition_name + "1_pkl.pkl", "rb") as f1:
              try:
                prod_mask = pkl.load(f1, encoding = "latin-1")
              except EOFError:
                print("pkl.load fail")
                break

            final_process(model_image, middle_composition_name, prod_mask['product_mask'],
                          model_mask['model_mask'], interval, FLAGS.isUpper, 0)

            #################################(#1 끝)#############################
            ##################(#2 시작 - final_upper 와 lower 합성)###############
            # 상의가 무조건 하의 덮게

            final_name = FLAGS.output + "/final_images/" + full_composition_name + "final.png"

            if FLAGS.lowerId == "000000": #상의만 있는 경우 shutil 활용 복사
              coarse_image = _load_image(FLAGS.output + "/final_upper_images/" + middle_composition_name + "final.png")
              final_image = final_trim_image(coarse_image)
              scipy.misc.imsave(final_name, final_image)
              #shutil.copy2(FLAGS.output + "/final_upper_images/"+middle_composition_name+"final.png", final_name )
            elif FLAGS.upperId == "000000": #하의만 있는 경우 shutil 활용 복사
              coarse_image = _load_image(FLAGS.output + "/final_lower_images/" + middle_composition_name + "final.png")
              final_image = final_trim_image(coarse_image)
              scipy.misc.imsave(final_name, final_image)
              #shutil.copy2(FLAGS.output + "/final_lower_images/"+middle_composition_name+"final.png", final_name )
            #두 이미지를 합쳐야 할 경우
            elif (FLAGS.lowerId != "000000") and (FLAGS.upperId != "000000"):

              composition_name = upper_composition_name
              coarse_image = _load_image(FLAGS.output + "/final_lower_images/" + lower_composition_name + "final.png")

              with open(FLAGS.result_dir_stage2 + composition_name + "1_pkl.pkl", "rb") as f3:
                try:
                  prod_mask = pkl.load(f3, encoding = "latin-1")
                except EOFError:
                  print("pkl.load fail")
                  break
              
              fg = prod_mask['product_mask']
              composition_raw = _load_image(FLAGS.result_dir_stage2 + composition_name + "final.png")
              composition_mask = _load_image(FLAGS.result_dir_stage2 + composition_name + "mask.png")


              # isUpper 값을 무조건 1로
              bg = save_model_mask(coarse_image, composition_raw, composition_mask,
                                   full_composition_name, interval,
                                   1, tag='save')
              
              final_process(coarse_image, full_composition_name, fg, bg, interval,1, 1)
              print("아직 아님!")


            post2 = time.time()
            print("[post_processing] process time : " + str(post2 - post1))



          

            #############통신##############
            result_dir_final = final_name
            print(result_dir_final)

            try:
              result_file = open(result_dir_final,'rb')
              upload = {'fileToUpload':result_file}
              user_id = FLAGS.userId
              upper_id = FLAGS.upperId
              lower_id = FLAGS.lowerId


              #obj = {'userid':image_name[:-5], 'productid':product_image_name[:-5]}
              obj = {'userid':user_id, 'upper':upper_id, 'lower':lower_id}
              print(obj)
              res = requests.post('http://211.253.229.68/get_res.php',files=upload,data=obj)
              print(res)
            except Exception as ex:
              print("전송 오류", ex)
    except Exception as ex:
      print("오류", ex)   
  
if __name__ == "__main__":
  tf.app.run()
