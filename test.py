import numpy as np 
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b



image = tf.zeros([4,2,3])

# Launch the graph in a session.
# sess = tf.compat.v1.Session()


with tf.Session() as sess:

   
    # Evaluate the tensor `c`.

    # ed = tf.expand_dims(tf.range(3),0)
    # print('expand dim')
    # print('expand dim original',sess.run(tf.range(3)))
    # print('expand dim 0',sess.run( tf.expand_dims(tf.range(3),0)))
    # print('expand dim 1',sess.run( tf.expand_dims(tf.range(3),1)))
    # print('expand dim -1',sess.run( tf.expand_dims(tf.range(3),-1)))


    T = 2
    N = 3
    E = 10
    # position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
    # print('position_ind')
    # print(sess.run(position_ind))
    # # print(sess.run(image)) # prints 30.0


    # decoder_inputs = tf.range(3)
    # tgt_masks = tf.math.equal(decoder_inputs, 0)
    # print(sess.run(tgt_masks)) # [ True False False]


    # tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    # print(sess.run(tf.ones_like(tensor))) 
    # # Given a single tensor (tensor), this operation returns a 
    # # tensor of the same type and shape as tensor with all elements set to 1

    # position encoding
    # ----
    # position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
    # # Constructs a tensor by tiling a given tensor.
    # maxlen = 5
    # print(sess.run(position_ind))
    # # First part of the PE function: sin and cos argument
    # position_enc = np.array([
    # [pos / np.power(10000, (i-i%2)/E) for i in range(E)] # d_model
    # for pos in range(maxlen)])
    # print(position_enc)

    # Second part, apply the cosine to even columns and sin to odds.
    # position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    # position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    # position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)
    # print(sess.run(position_enc))

    # lookup
    # outputs = tf.nn.embedding_lookup(position_enc, position_ind)


    x = tf.random.uniform([2, 30], -1, 1)
    x = tf.constant([[1, 2, 3,2,3,5], [4, 5, 6,3,4,5]]) # 2d
    x = tf.constant([[[1, 2,2], [3,2,3]], [[4, 5,4], [6,3,5]]]) # 3d
    print(sess.run(x))
    # # print(sess.run(x))
    # s0, s1, s2,s3,s4,s5 = tf.split(x, num_or_size_splits=6, axis=1)
    # print(sess.run(s0))
    # print(sess.run(tf.split(Q, num_heads, axis=2)))
    # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)

    # num_heads = 3
    # Q = tf.expand_dims(tf.constant([[1, 2, 3], [4, 5, 6]]),1)
    # print(sess.run(Q))
    # print(sess.run(tf.split(Q, num_heads, axis=2)))
    # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    # print(sess.run(Q_)) # [ True False False]


    # zeros_initializer
    # inputs_shape = x.get_shape()
    # print(inputs_shape)
    # params_shape = inputs_shape[-1:]
    # print(params_shape)
    # beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    # sess.run(tf.global_variables_initializer())
    # print(sess.run(beta))


    # xavier_initializer
    # Draws samples from a truncated normal distribution centered on 0
    # with stddev = sqrt(2 / (fan_in + fan_out)) 
    # where fan_in is the number of input units in the weight 
    # tensor and fan_out is the number of output units in the weight tensor.
    vocab_size = 5
    num_units = 64
    embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
    sess.run(tf.global_variables_initializer())
    print(sess.run(embeddings))


    