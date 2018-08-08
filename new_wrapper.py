import nst_utils2
import tensorflow as tf
import cv2

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
    J_style_layer = (1./(4 * (n_C*n_H*n_W)**2)) * tf.reduce_sum(tf.pow((GS - GG), 2))
    return J_style_layer

# def compute_style_cost(model, STYLE_LAYERS, sess, out):
#     J_style = 0
#     idx=0
#     while idx<len(STYLE_LAYERS):
#         layer_name = STYLE_LAYERS[idx][0]
#         coeff = STYLE_LAYERS[idx][1]
#         a_S = sess.run(out)
#         a_G = out
#         J_style_layer = compute_layer_style_cost(a_S, a_G)
#         J_style += coeff * J_style_layer
#         idx=idx+1
#     return J_style

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J


# def model_nn(sess, input_image, output_file, num_iterations = 100):
#     sess.run(tf.global_variables_initializer())
#     sess.run(model['input'].assign(input_image))
#     for i in range(num_iterations):
#         sess.run(train_step)
#         generated_image = sess.run(model['input'])
#         if i%20 == 0:
#             Jt, Jc, Js = sess.run([J, J_content, J_style])
#             print("Iteration " + str(i) + " :")
#             print("total cost = " + str(Jt))
#             print("content cost = " + str(Jc))
#             print("style cost = " + str(Js))
#             save_image(str(i) + str(random.random()) +".png", generated_image)
#
#     # save last generated image
#     save_image(output_file, generated_image)
#
#     return


def stylize(content_file = 'alia.jpg', style_file = 'van_gogh.jpg', outfile = 'outfile.jpg', num_iterations = 100):
    content_image = cv2.imread(content_file)
    content_image = nst_utils2.reshape_and_normalize_image(content_image)
    style_image = cv2.imread(style_file)
    con_img_shape = content_image[0].shape
    style_image = cv2.resize(style_image, (con_img_shape[1], con_img_shape[0]))
    style_image = nst_utils2.reshape_and_normalize_image(style_image)
    generated_image = nst_utils2.generate_noise_image(content_image)

    tf.reset_default_graph()
    global sess
    sess = tf.InteractiveSession()
    model = nst_utils2.load_vgg_model("imagenet-vgg-verydeep-19.mat", content_image)
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))
    STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    for i in range(num_iterations):
        print('Running Iteration', i)
        # tf.reset_default_graph()
        # sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(generated_image))
        _  = sess.run(train_step)
        generated_image = sess.run(model['input'])
        generated_image_to_save = generated_image[0]
        if i%20 < 21:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            cv2.imwrite(str(i) +".jpg", generated_image_to_save)
    cv2.imwrite(outfile, generated_image_to_save)

if __name__ == '__main__':
    stylize()
# content_file = 'alia.jpg'
# style_file = 'van_gogh.jpg'
