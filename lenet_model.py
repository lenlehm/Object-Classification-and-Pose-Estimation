import tensorflow as tf
# LeNet Architecture
# Conv (64*64*C)
# - in_channel: C, out_channel: 16, filter_size: 8
# Relu
# MaxPool(2*2)
# Conv (28*28*16)
# - in_channel: 16, out_channel: 7, filter_size: 5
# Relu
# MaxPool(2*2)
# FC Conv (12*12*7) -> 256
# FC Output descriptor size: 16



def leNet(features, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 64, 64, 3])

    with tf.name_scope("LeNet"):

        # Convolutional Layer #1
        end_point = 'conv1_57x57x16'
        net = tf.layers.conv2d(
          inputs=input_layer,
          filters=16,
          kernel_size=[8, 8],
          activation=tf.nn.relu,
          name=end_point)

        # Pooling Layer #1
        end_point = 'pool1_28x28x16'
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

        # Convolutional Layer #2 and Pooling Layer #2
        end_point = 'conv2_24x24x7'
        net = tf.layers.conv2d(
          inputs=net,
          filters=7,
          kernel_size=[5, 5],
          activation=tf.nn.relu, name=end_point)

        # Pooling Layer #2
        end_point = 'poo2_12x12x7'
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

        # Dense Layer
        net = tf.reshape(net, [-1, 1008])
        end_point = 'dense_256'
        net = tf.layers.dense(inputs=net, units=256, activation=None, name=end_point)

        # Logits Layer
        end_point = 'logits_16'
        logits = tf.layers.dense(inputs=net, units=16, activation=None, name=end_point)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "descriptors": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    batch_size = len(features)

    diff_pos = logits[0:batch_size:3] - logits[1:batch_size:3]
    diff_neg = logits[0:batch_size:3] - logits[2:batch_size:3]
    l2_loss_diff_pos = tf.nn.l2_loss(diff_pos) * 2
    l2_loss_diff_neg = tf.nn.l2_loss(diff_neg) * 2

    m = 0.01
    loss_triplets = tf.reduce_sum(tf.maximum(0., (1.-(l2_loss_diff_neg/(l2_loss_diff_pos+m)))))

    loss_pairs = tf.nn.l2_loss(l2_loss_diff_pos) * 2

    loss = loss_triplets + loss_pairs

    # Creates summaries for output node
    tf.summary.scalar(family='Training loss', tensor=loss, name="total_loss")
    tf.summary.scalar(family='Training loss', tensor=loss_triplets, name="loss_triplets")
    tf.summary.scalar(family='Training loss', tensor=loss_pairs, name="loss_pairs")
    tf.summary.merge_all()

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        # writer = tf.summary.FileWriter('./LeNet_graph', sess.graph)
        # writer.close()
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    # eval_metric_ops = {
    #     "accuracy": tf.metrics.accuracy(
    #         labels=labels, predictions=predictions["classes"])}
    eval_metric_ops = {
    "accuracy": 0}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
