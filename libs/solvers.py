from tensorflow.python.training.adam import AdamOptimizer

def optimizer(params, cost, global_step, name="adam" ):
    adam = AdamOptimizer( learning_rate=params.lrate, name=name )
    return adam.minimize( cost, global_step=global_step)
