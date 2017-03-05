# https://habrahabr.ru/post/305578/
import tensorflow as tf

if __name__ == '__main__':
    print '--------Start-------'
    graph = tf.get_default_graph()
    print graph.get_operations()
    input_value = tf.constant(1.0)
    operations = graph.get_operations()
    sess = tf.Session()
    # print sess.run(input_value)

    weight = tf.Variable(0.8)

    output_value = weight * input_value
    op = graph.get_operations()[-1]

    # for op_input in op.inputs: print op_input
    init = tf.global_variables_initializer()
    sess.run(init)
    print sess.run(output_value)

    x = tf.constant(1.0, name='input')
    w = tf.Variable(0.8, name='weight')
    y = tf.multiply(w, x, name='weight')

    y_ = tf.constant(0.0)
    loss = (y-y_)**2
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)

    grads_and_vars = optim.compute_gradients(loss)
    sess.run(tf.global_variables_initializer())
    sess.run(optim.apply_gradients(grads_and_vars))
    # print sess.run(w)
    train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
    for i in range(100):
        print('before step {}, y is {}'.format(i, sess.run(y)))
        sess.run(train_step)

    print '--------End-------'
