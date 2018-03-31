import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from nst_utils import save_image, reshape_and_normalize_image, load_vgg_model, generate_noise_image




STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C, [n_C, n_H*n_W])
    a_G_unrolled = tf.reshape(a_G, [n_C, n_H*n_W])
    
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))/(4*n_H*n_W*n_C)
    return J_content


def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, [-1,n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [-1,n_C]))
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/np.square(2*n_H*n_W)
    return np.sum(J_style_layer)


def style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, lm in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = layer_style_cost(a_S, a_G)
        J_style += lm * J_style_layer
    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content + beta*J_style
    return J



################################################################################
model = load_vgg_model('./model/imagenet-vgg-verydeep-19.mat')

sess = tf.InteractiveSession()
style_image =scipy.misc.imread( 'images/style_2.jpg')
style_image = reshape_and_normalize_image(style_image)

content_image = scipy.misc.imread('images/content_2.jpg')
content_image = reshape_and_normalize_image(content_image)

generated_image = generate_noise_image(content_image)
#plt.imshow(generated_image.reshape((300,400,3)))
#model = load_vgg_model('./model/imagenet-vgg-verydeep-19.mat')

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = content_cost(a_C, a_G)


sess.run(model['input'].assign(style_image))
J_style = style_cost(model, STYLE_LAYERS)

J = total_cost(J_content, J_style, 10, 40)

optimizer = tf.train.AdamOptimizer(learning_rate=1.5)
train_step = optimizer.minimize(J)

def nn_model(sess, input_image, num_iterations=300):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            
            save_image("./a_" + str(i) + ".png", generated_image)
    
    
    save_image('./generated_image_2.jpg', generated_image)
    
    return generated_image

nn_model(sess, generated_image)

