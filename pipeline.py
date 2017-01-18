from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def extract( features, name='digit1' ):
    return tf.cast( features[ name ], tf.int64 )


def get_label( feature ):
    return [ extract( feature, name='digit{}'.format( digit ) ) for digit in range( 0, 6 ) ]


def image_distortions( image ):
    with tf.name_scope("RandomDistortions"):
        image = tf.image.random_saturation( image, lower=0.5, upper=1.5, seed=99 )
        image = tf.image.random_brightness( image, max_delta=64. / 255., seed=99 )
        image = tf.image.random_hue( image, max_delta=0.2, seed=99 )
        image = tf.image.random_contrast( image, lower=0.5, upper=1.5, seed=99 )
        return image


def read_and_decode( filename_queue, params ):
    with tf.name_scope("Parse"):
        reader = tf.TFRecordReader( )
        _, serialized_example = reader.read( filename_queue )
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_raw': tf.FixedLenFeature( [ ], tf.string ),
                    'height': tf.FixedLenFeature( [ ], tf.int64 ),
                    'width': tf.FixedLenFeature( [ ], tf.int64 ),
                    'depth': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit1': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit2': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit3': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit4': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit5': tf.FixedLenFeature( [ ], tf.int64 ),
                    'digit0': tf.FixedLenFeature( [ ], tf.int64 ),
                } )
        
        # Decode images to uint8
        image = tf.decode_raw( features[ 'image_raw' ], tf.uint8 , name="Decode")
        
        # Set image shape to dense format
        image.set_shape( [ 64 * 64 * 3 ] )
    
    # Reshape image from dense to (H, W, C)
    with tf.name_scope("Reshape"):
        image = tf.reshape( image, [ 64, 64, 3 ] )

    # Apply distortions
    if params.is_training:
        image = image_distortions( image )
        if params.random_crop:
            with tf.name_scope( "RandomCrop" ):
                image = tf.random_crop( image, [ 54, 54, params.channel ] )
        
    # Convert to gray scale
    if params.grayscale:
        with tf.name_scope("GrayScale"):
            image = tf.image.rgb_to_grayscale( image )

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = get_label( features )
        
    # Resize image into 32x32x3
    with tf.name_scope("Resize"):
        image = tf.image.resize_images( image, [ 32, 32 ] )
    
    # Zero Mean (Normalization)
    with tf.name_scope( "Normalize" ):
        image = tf.sub( image, tf.reduce_mean( image ) )

    return image, label


def inputs( params ):
    
    filename = params.records_dir + ".tfrecords"
    
    with tf.variable_scope( 'Images' ) as scope:
        filename_queue = tf.train.string_input_producer(
                [ filename ], num_epochs=params.num_epochs, name="Queue")
        
        image, label = read_and_decode( filename_queue, params)
        
        images, sparse_labels = tf.train.shuffle_batch( [ image, label ],
                                                        batch_size=params.batch_size,
                                                        num_threads=params.num_threads,
                                                        seed=99,
                                                        capacity=1000 + 3 * params.batch_size,
                                                        min_after_dequeue=1000, name="Shuffle" )
        
        return images, sparse_labels


if __name__ == '__main__':
    pass