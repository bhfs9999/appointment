import tensorflow as tf

a = tf.random_normal((10000, 10000))
b = tf.random_normal((10000, 10000))
c = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(c))
