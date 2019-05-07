"""
Wrap the tf.estimator.Estimator API.
That way, we can add separate profiling annotations to session.run(...) calls for:
- train(..., hooks=..., saving_listeners=...): A call to train_op that performs a forward pass + gradient update.
- predict(..., hooks=...): A forward pass / inference.
- evaluate(..., hooks=...): A call for computing validation metrics.
- export_saved_model(...): A call for saving a model(?)

IMPLEMENTATION:
In order to wrap these session.run(...) calls, we use the "hooks" callback API for train/predict/evaluate,
and the "saving_listeners" for model-checkpointing.
In particular, the train/predict/evaluate methods simply append our wrapper to the end of the hooks/saving_listeners arguments.

NOTE: We don't bother to wrap export_saved_model, simply because it doesn't expose any convenient hooks...
but we COULD wrap it (minigo doesn't use it).
"""

import tensorflow as tf

from iml_profiler.profiler import glbl

from tensorflow.python.training.training import CheckpointSaverListener

"""
Wrap tf.estimator.Estimator.
"""
old_Estimator = None
def setup_wrap_Estimator():
    global old_Estimator
    old_Estimator = tf.estimator.Estimator
    tf.estimator.Estimator = EstimatorWrapper

SETUP_DONE = False
def setup():
    global SETUP_DONE
    assert not SETUP_DONE
    setup_wrap_Estimator()
    SETUP_DONE = True

class EstimatorWrapper:
    def __init__(self, *args, **kwargs):
        # Explicitly wrap only certain methods.
        #
        # We do this instead of inheritting from tf.estimator.Estimator.
        # Would rather fail on an un-wrapped method, than silently continue unknowingly
        # without a set_operation/end_operation call.
        self.estimator = old_Estimator(*args, **kwargs)

    def latest_checkpoint(self):
        return self.estimator.latest_checkpoint()

    def train(self,
              input_fn,
              hooks=None,
              steps=None,
              max_steps=None,
              saving_listeners=None):

        if hooks is None:
            hooks = []
        hooks = hooks + [TrainEstimatorHook(glbl.prof)]

        if saving_listeners is None:
            saving_listeners = []
        saving_listeners = saving_listeners + [SaveModelEstimatorHook(glbl.prof)]

        return self.estimator.train(
            input_fn,
            hooks=hooks,
            steps=steps,
            max_steps=max_steps,
            saving_listeners=saving_listeners,
        )

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_examples=True):

        if hooks is None:
            hooks = []
        hooks = hooks + [PredictEstimatorHook(glbl.prof)]

        return self.estimator.predict(
            input_fn,
            predict_keys,
            hooks,
            checkpoint_path,
            yield_single_examples)

    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None,
                 name=None):

        if hooks is None:
            hooks = []
        hooks = hooks + [EvaluateEstimatorHook(glbl.prof)]

        return self.estimator.evaluate(
            input_fn,
            steps,
            hooks,
            checkpoint_path,
            name)

    def export_savedmodel(self, export_dir_base, serving_input_receiver_fn,
                          assets_extra=None,
                          as_text=False,
                          checkpoint_path=None,
                          strip_default_attrs=False):
        raise NotImplementedError("IML: Haven't bothered to wrap tf.estimator.Estimator.export_savedmodel with profiling set_operation/end_operation annotations.")

    def export_saved_model(self,
                           export_dir_base,
                           serving_input_receiver_fn,
                           assets_extra=None,
                           as_text=False,
                           checkpoint_path=None):
        # TensorFlow 2.0 I think;
        # export_savedmodel from 1.0 is deprecated.
        raise NotImplementedError("IML: Haven't bothered to wrap tf.estimator.Estimator.export_saved_model with profiling set_operation/end_operation annotations.")

class ProfileEstimatorHook(tf.train.SessionRunHook):
    def __init__(self, operation, prof=None):
        """
        Hook for Estimator callback API to add profiling set_operation/end_operation
        annotations before/after invoking TensorFlow session.run(...) calls.

        :param operation:
            The name of the operation to use for prof.set_operation(...).
        """
        self.operation = operation
        if prof is None:
            self.prof = glbl.prof
        else:
            self.prof = prof
        assert self.prof is not None

    # def begin(self):
    #     pass

    def before_run(self, run_context):
        self.prof.set_operation(self.operation)

    def after_run(self, run_context, run_values):
        self.prof.end_operation(self.operation)

    # def end(self, session):
    #     pass

class SaveModelEstimatorHook(CheckpointSaverListener):
    def __init__(self, prof=None):
        self.operation = 'estimator_save_model'
        if prof is None:
            self.prof = glbl.prof
        else:
            self.prof = prof
        assert self.prof is not None

    # def begin(self):
    #     pass

    def before_save(self, session, global_step_value):
        self.prof.set_operation(self.operation)

    def after_save(self, session, global_step_value):
        self.prof.end_operation(self.operation)

    # def end(self, session, global_step_value):
    #     pass

class TrainEstimatorHook(ProfileEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_train', prof=prof)
class PredictEstimatorHook(ProfileEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_predict', prof=prof)
class EvaluateEstimatorHook(ProfileEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_evaluate', prof=prof)
