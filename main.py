from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from tqdm import tqdm
from glob import glob

from config import SVHNParams
from train import trainer
from evaluate import evaluator


def main(params):
    print("\nMULTI-SVHN CLASSIFICATION")
    print("Running project: {}".format(params.name))
    print("Mode: {}".format(params.mode))
    print("Is training: {}".format(params.is_training))
    print("Logger: {}".format(params.log))
    print("Checkpoint dir: {}".format(params.checkpoint_dir))
    print("Loading data from {}".format(params.record_name[1:]))
    print("Num epochs: {}".format(params.num_epochs))
    print("Batch size: {}".format(params.batch_size))
    print("Learning rate: {}".format(params.lrate))
    print("Random crop: {}".format(params.random_crop))
    print("Grayscale: {}".format(params.grayscale))
    print("Channels: {}\n".format(params.channel))

    # pprint( vars( params ) )

    if params.mode == "train":

        trainer(params)

    elif params.mode == "test":

        running_results = {
            'checkpoint'       : [],
            'sequence_accuracy': [],
            'digit_accuracy'   : []
        }

        # Collect checkpoints to evaluate
        checkpoints_to_eval = glob('{}*.meta'.format(params.checkpoint_dir))

        # Remove "meta" extension
        checkpoint_list = [chkpt[:-5] for chkpt in checkpoints_to_eval]

        # Take the step number (last number index)
        checkpoints_index = [int(i.split('-')[-1]) for i in checkpoint_list]

        # Run an evaluation for each checkpoint
        for index, checkpoint in tqdm(zip(checkpoints_index, checkpoint_list)):
            results = evaluator(params, checkpoint)
            running_results['checkpoint'].append(index)
            running_results['sequence_accuracy'].append(results['sequence_accuracy'][0])
            running_results['digit_accuracy'].append(results['digit_accuracy'][0])

        # Create a data frame with results
        df = pd.DataFrame(running_results)
        results = df.sort_values(by='checkpoint', ascending=True)

        # Save results to results directory
        results.to_csv(params.results_path, index=False)
        print("Results saved to {}\n".format(params.results_path))

        # Preview results in terminal
        print(results)

    elif params.mode == "valid":
        running_results = {
            'checkpoint'       : [],
            'sequence_accuracy': [],
            'digit_accuracy'   : []
        }

        # Collect checkpoints to evaluate
        checkpoints_to_eval = glob('{}*.meta'.format(params.checkpoint_dir))

        # Remove "meta" extension
        checkpoint_list = [chkpt[:-5] for chkpt in checkpoints_to_eval]

        # Take the step number (last number index)
        checkpoints_index = [int(i.split('-')[-1]) for i in checkpoint_list]

        # Run an evaluation for each checkpoint
        for index, checkpoint in tqdm(zip(checkpoints_index, checkpoint_list)):
            results = evaluator(params, checkpoint)
            running_results['checkpoint'].append(index)
            running_results['sequence_accuracy'].append(results['sequence_accuracy'][0])
            running_results['digit_accuracy'].append(results['digit_accuracy'][0])
            print()

        # Create a data frame with results
        df = pd.DataFrame(running_results)
        results = df.sort_values(by='checkpoint', ascending=True)

        # Save results to results directory
        results.to_csv(params.results_path, index=False)
        print("Results saved to {}\n".format(params.results_path))

        # Preview results in terminal
        print(results)


if __name__ == '__main__':
    params = SVHNParams()
    main(params)
