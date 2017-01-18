import os
import tensorflow as tf
import itertools
from tqdm import tqdm
import numpy as np


def _int64_feature( value, index=None ):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    if index:
        return tf.train.Feature( int64_list=tf.train.Int64List( value=[ value[ index ] ] ) )
    else:
        return tf.train.Feature( int64_list=tf.train.Int64List( value=[ value ] ) )


def _bytes_feature( value ):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature( bytes_list=tf.train.BytesList( value=[ value ] ) )


# Load data from disk
def load_data_from(path, name):
    assert(name != None)
    save = np.load( path )
    print(save.keys())
    data = save['{}_dataset'.format(name)]
    labels = save['{}_labels'.format(name)]
    del save
    
    print( "{} shape: {}\n".format( name, data.shape ) )
    
    return data, labels

def convert_to_records( path, name, limit=True ):
    """Converts a dataset to tfrecords."""
    
    images, labels = load_data_from(path, name)
    
    images = images.astype( np.uint8 )
    
    num_examples = images.shape[ 0 ]
    
    if images.shape[ 0 ] != num_examples:
        raise ValueError( 'Images size %d does not match label size %d.' %
                          (images.shape[ 0 ], num_examples) )
    
    rows = images.shape[ 1 ]
    cols = images.shape[ 2 ]
    depth = images.shape[ 3 ]
    
    filename = os.path.join( './records/{}'.format( name + '.tfrecords' ))
    
    print( 'Writing', filename )
    writer = tf.python_io.TFRecordWriter( filename )
    for index in tqdm( range( num_examples ) ):
        digit0 = int( labels[ index ][ 0 ] )
        digit1 = int( labels[ index ][ 1 ] )
        digit2 = int( labels[ index ][ 2 ] )
        digit3 = int( labels[ index ][ 3 ] )
        digit4 = int( labels[ index ][ 4 ] )
        digit5 = int( labels[ index ][ 5 ] )
        image_raw = images[ index ].tostring( )
        example = tf.train.Example( features=tf.train.Features( feature={
            'height': _int64_feature( rows ),
            'width': _int64_feature( cols ),
            'depth': _int64_feature( depth ),
            'digit1': _int64_feature( digit1 ),
            'digit2': _int64_feature( digit2 ),
            'digit3': _int64_feature( digit3 ),
            'digit4': _int64_feature( digit4 ),
            'digit5': _int64_feature( digit5 ),
            'digit0': _int64_feature( digit0 ),
            'image_raw': _bytes_feature( image_raw )
        }
        ) )
        writer.write( example.SerializeToString( ) )
    writer.close( )


if __name__ == '__main__':
    path_to_dir = os.path.abspath( '.' )
    train_path = path_to_dir + "/svhn/interim/train.npz"
    test_path = path_to_dir + "/svhn/interim/test.npz"
    valid_path = path_to_dir + "/svhn/interim/valid.npz"
    convert_to_records(train_path, 'train' )
    convert_to_records(test_path, 'test' )
    convert_to_records(valid_path, 'valid' )