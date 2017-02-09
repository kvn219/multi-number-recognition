from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from pipeline import inputs
from libs.architecture import inference
from libs.utilities import *


def evaluator(params, checkpoint):
    with tf.Graph().as_default() as graph:

        # Inputs
        images, labels = inputs(params)

        # Model
        logits = inference(params, images)

        # Predictions
        predictions = predict(logits)

        scores = score_digits_in_image(predictions)

        correct = correct_digits_in_image(scores, labels)

        seq_acc = sequence_accuracy(predictions, labels)

        dig_acc = digit_accuracy(correct, labels)

        # Initializer
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer(),
                        name="initializer")

        with tf.Session(graph=graph) as sess:

            # Start  session
            sess.run(init)

            # Saver
            saver = tf.train.Saver(max_to_keep=500, name="Saver")

            # Coordinator
            coord = tf.train.Coordinator()

            # Treads
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Restore model
            print("\nRestoring...\n", checkpoint)
            saver.restore(sess, checkpoint)

            start_time = datetime.now()
            results = {
                'checkpoint'       : [],
                'sequence_accuracy': [],
                'digit_accuracy'   : []
            }

            try:
                step = 0

                while not coord.should_stop():
                    seq_score, dig_score, y_pred, X_test, y_true = sess.run(
                            [seq_acc, dig_acc, predictions, images, labels])
                    results['sequence_accuracy'].append(seq_score)
                    results['digit_accuracy'].append(dig_score)
                    results['checkpoint'].append(checkpoint)
                    step += 1

            except tf.errors.OutOfRangeError:
                print('Stopping evaluation at {:4d} epochs, {:3d} steps.'.format(params.num_epochs, step))

            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()
            print("Total time to run: {}".format(datetime.now() - start_time))
            return results
