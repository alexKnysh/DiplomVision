import pybrain
import tensorflow as tf

if __name__ == '__main__':
    filenames = ['im_01.jpg']
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_jpeg(content, channels=3)
    image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(image, [224, 224])
    image_batch = tf.train.batch([resized_image], batch_size=8)
