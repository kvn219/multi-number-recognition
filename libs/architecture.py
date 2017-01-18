import tensorflow as tf
from libs.layers import convolution, affine, connect, dropout



def inference( params, images ):
    with tf.variable_scope( 'ConvNet' ):
        conv = convolution( images, (params.channel, 16), filter=3, stride=1, name='C1', batch_norm=True, mode=params.mode, verbose=params.verbose )
        conv = convolution( conv, (16, 32), filter=3, stride=1, name='C2', batch_norm=True, mode=params.mode, verbose=params.verbose )
        conv = convolution( conv, (32, 64), filter=5, stride=3, name='C3', batch_norm=True, mode=params.mode, verbose=params.verbose )
        conv = convolution( conv, (64, 256), filter=5, stride=3, name='C4', batch_norm=True, mode=params.mode, verbose=params.verbose )
        
        if params.mode == "train":
            feature_map = dropout(conv, .85, name="D5")
        
        else:
            feature_map = conv
        
        dense = affine( feature_map, 512, name="FC6", is_last=True )
        
        digits = [ connect( dense, 11, name='Digit{}'.format( i )) for i in range( 1, 6 ) ]
    
    return digits