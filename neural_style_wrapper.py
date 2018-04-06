import os
os.chdir('/home/ec2-user/neural-style')
import sys
import scipy.io
import scipy.misc
from PIL import Image
from nst_utils2 import *
import numpy as np
import tensorflow as tf
import random

def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(a_C)
    a_G_unrolled = tf.transpose(a_G)
    J_content = (1/ (4* n_H * n_W * n_C)) * tf.reduce_sum(tf.pow((a_G_unrolled - a_C_unrolled), 2))
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = (1./(4 * n_C**2 * (n_H*n_W)**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS, sess, out):
    J_style = 0
    idx=0
    print 'check 511'
    while idx<len(STYLE_LAYERS):
        print 'check 512 for idx' + str(idx)
        layer_name = STYLE_LAYERS[idx][0]
        coeff = STYLE_LAYERS[idx][1]
        print 'check 513 for idx' + str(idx)
        a_S = sess.run(out)
        print 'a_S or a_G??'
        a_G = out
        print 'check 514 for idx' + str(idx)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
        idx=idx+1
        print 'check 515 for idx' + str(idx)
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J


def model_nn(sess, input_image, output_file, num_iterations = 100):
    print 'check_9'
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    print 'check_10'
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image(str(i) + str(random.random()) +".png", generated_image)
    print 'check_11'
    
    # save last generated image
    save_image(output_file, generated_image)
    print 'check_12'
    
    return


def stylize(content_file, style_file, outfile):
    #global sess
    #global out

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    #content_file = 'alia.jpg'
    #style_file = 'van_gogh.jpg'
    print 'check_1'
    content_image_orig = scipy.misc.imread(content_file)
    content_image = reshape_and_normalize_image(content_image_orig)
    #style_image = scipy.misc.imread(style_file)
    style_image = reshaped_style_image(style_file, content_image_orig)
    style_image = reshape_and_normalize_image(style_image)
    generated_image = generate_noise_image(content_image)
    print 'check_2'

    model = load_vgg_model("imagenet-vgg-verydeep-19.mat", content_image)
    print 'check_3'

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    print 'check_4'

    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)
    print 'check_5'

    sess.run(model['input'].assign(style_image))
    print 'check_51'
    STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
    #J_style = compute_style_cost(model, STYLE_LAYERS, sess, out)

    J_style = 0
    idx=0
    print 'check 511'
    while idx<len(STYLE_LAYERS):
        print 'check 512 for idx' + str(idx)
        layer_name = STYLE_LAYERS[idx][0]
        coeff = STYLE_LAYERS[idx][1]
        print 'check 513 for idx' + str(idx)
        a_S = sess.run(out)
        print 'a_S or a_G??'
        a_G = out
        print 'check 514 for idx' + str(idx)
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
        idx=idx+1
        print 'check 515 for idx' + str(idx)


    print 'check_52'
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    print 'check_6'


    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    print 'check_7'

    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated_image))
    print 'check_8'

    model_nn(sess = sess, input_image = generated_image, output_file = outfile, num_iterations = 100)
    print 'check_final'
