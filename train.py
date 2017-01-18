from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pipeline import inputs
from libs.architecture import inference
from libs.utilities import *
from libs.solvers import optimizer


def trainer(params):
    
    with tf.Graph( ).as_default( ) as graph:
        
        # Inputs
        with tf.name_scope("Pipeline"):
            global_step = tf.Variable( 0, trainable=False, dtype=tf.int32 , name="GlobalStep")
            images, labels = inputs( params )
            tf.summary.image( "Incoming", images, max_outputs=10 )
        
        # Model
        with tf.name_scope("Inference"):
            logits = inference(params, images)
        
        # Loss
        with tf.name_scope("Cost"):
            cost = error( logits, labels )
            train_op = optimizer( params, cost, global_step, name="Adam" )
            
        # Predictions
        with tf.name_scope("Predictions"):
            predictions = predict( logits )
            scores = score_digits_in_image( predictions )
            correct = correct_digits_in_image( scores, labels )

        with tf.name_scope("Metrics/Accuracy"):
            seq_acc = sequence_accuracy(predictions, labels)
            dig_acc = digit_accuracy(correct, labels)
        
        ## Initializer
        init = tf.group( tf.global_variables_initializer( ),
                         tf.local_variables_initializer( ),
                         name="Initializer" )
        
        # add summary op
        tf.summary.scalar( 'Loss', cost)
        tf.summary.scalar('DigitAccuracy', dig_acc)
        tf.summary.scalar('SequenceAccuracy', seq_acc)

        for var in tf.trainable_variables( ):
            tf.summary.histogram( var.op.name, var )

        summary_op = tf.summary.merge_all( )
        
        with tf.Session( graph=graph ) as sess:
            
            # Start  session
            sess.run( init )
            
            # TensorBoard
            summary_writer = tf.summary.FileWriter( "./logs/{}/{}".format(params.log, params.name), graph=tf.get_default_graph( ) )
            
            # Saver
            saver = tf.train.Saver( max_to_keep=500, name="Saver")
            
            # Coordinator
            coord = tf.train.Coordinator( )
            
            # Treads
            threads = tf.train.start_queue_runners( sess=sess, coord=coord )
            
            # Path to checkpoints
            checkpoint = tf.train.get_checkpoint_state( "./checkpoints/{}/".format(params.name))
            
            # Restore latest checkpoint, if there is one
            if checkpoint and checkpoint.model_checkpoint_path:
                print( "\nRestored model from: {}".format( checkpoint.model_checkpoint_path ) )
                saver.restore( sess, checkpoint.model_checkpoint_path)
            
            else:
                print( "Starting new {} session".format( params.mode ) )
                step = 0
            
            print( "Training...\n" )
            
            # Checkpoint save path
            checkpoint_save_path = params.checkpoint_dir + "{}".format(params.name)
            
            start_time = datetime.now( )
            
            try:
                while not coord.should_stop( ):
                    
                    _, step = sess.run( [train_op, global_step] )

                    if step % 1000 == 0:
                        d_acc, s_acc, loss, summary = sess.run( [dig_acc, seq_acc, cost, summary_op] )
                        print( eval_fmt.format( datetime.now( ), step, loss, s_acc, d_acc ) )
                        saver.save( sess, checkpoint_save_path, global_step=step )
                        summary_writer.add_summary( summary, global_step= step )

                        
            
            except tf.errors.OutOfRangeError:
                print( 'Done training for {:4d} epochs, {:3d} steps.'.format( params.num_epochs, step ) )
            
            finally:
                saver.save( sess, checkpoint_save_path, global_step=step )
                coord.request_stop( )
                coord.join( threads )
            sess.close( )
            
        print("Finished training.")
    print( "Total time to run: {}".format( datetime.now( ) - start_time ) )
    