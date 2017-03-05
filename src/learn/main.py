import tensorflow as tf


if __name__ == '__main__':
    print '--------Start-------'
    graph = tf.get_default_graph()
    print graph.get_operations()
    input_value = tf.constant(1.0)
    operations = graph.get_operations()
    sess = tf.Session()
    print sess.run(input_value)
    print '--------End-------'