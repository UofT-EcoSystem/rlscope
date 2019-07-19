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

import iml_profiler

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
            # TODO: separate hook into 2 hooks, one BEFORE, one AFTER
            # if iml.prof.

        # Remove EstimatorReportProgressHook from hooks list.
        #
        # We need to ensure report_progress gets called BEFORE any set_operation calls,
        # and AFTER any end_operation calls.
        i = 0
        report_progress_hook = None
        while i < len(hooks):
            if isinstance(hooks[i], EstimatorReportProgressHook):
                report_progress_hook = hooks[i]
                del hooks[i]
                break
            i += 1
        for hook in hooks:
            assert not isinstance(hook, EstimatorReportProgressHook)

        iml_before_hooks = []
        if report_progress_hook is not None:
            iml_before_hooks.append(BeforeRunEstimatorReportProgressHook(report_progress_hook))
        iml_before_hooks.append(SetOpTrainEstimatorHook())

        iml_after_hooks = []
        iml_after_hooks.append(EndOpTrainEstimatorHook())
        if report_progress_hook is not None:
            iml_after_hooks.append(AfterRunEstimatorReportProgressHook(report_progress_hook))

        hooks = iml_before_hooks + hooks + iml_after_hooks

        if saving_listeners is None:
            saving_listeners = []
        saving_listeners = [SetOpSaveEstimatorHook()] + saving_listeners + [EndOpSaveEstimatorHook()]

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
        hooks = [SetOpPredictEstimatorHook()] + hooks + [EndOpPredictEstimatorHook()]

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
        hooks = [SetOpEvaluateEstimatorHook()] + hooks + [EndOpEvaluateEstimatorHook()]

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

class EstimatorReportProgressHook:
    """
    Hook for Estimator training callback that reports progress updates before each training time-step
    (just like what the stable-baselines training loops do manually.)

    NOTE: you need to add this manually to your Estimator.train(hooks=[...]) call.
    """
    def __init__(self, total_timesteps, prof=None):
        self.total_timesteps = total_timesteps
        self.is_running = False
        self.timestep = 0
        if prof is None:
            self.prof = iml_profiler.api.prof
        else:
            self.prof = prof
        assert self.prof is not None

    def before_run(self, run_context):
        # NOTE: iml.prof.enable_tracing() should be called externally
        self.is_running = True
        self._report_progress()

    def _report_progress(self):
        percent_complete = self.timestep / self.total_timesteps
        self.prof.report_progress(percent_complete,
                                  num_timesteps=self.timestep,
                                  total_timesteps=self.total_timesteps)

    def after_run(self, run_context, run_values):
        self.is_running = False
        self.timestep += 1
        # To avoid missing the last set of traced events when --iml-trace-time-sec isn't used.
        self._report_progress()

class BeforeRunEstimatorReportProgressHook(tf.train.SessionRunHook):
    def __init__(self, hook):
        assert type(hook) == EstimatorReportProgressHook
        self.hook = hook

    def before_run(self, run_context):
        self.hook.before_run(run_context)

class AfterRunEstimatorReportProgressHook(tf.train.SessionRunHook):
    def __init__(self, hook):
        assert type(hook) == EstimatorReportProgressHook
        self.hook = hook

    def after_run(self, run_context, run_values):
        self.hook.after_run(run_context, run_values)

# class ProfileEstimatorHook(tf.train.SessionRunHook):
#     def __init__(self, operation, prof=None):
#         """
#         Hook for Estimator callback API to add profiling set_operation/end_operation
#         annotations before/after invoking TensorFlow session.run(...) calls.
#
#         :param operation:
#             The name of the operation to use for prof.set_operation(...).
#         """
#         self.is_running = False
#         self.operation = operation
#         if prof is None:
#             self.prof = iml_profiler.api.prof
#         else:
#             self.prof = prof
#         assert self.prof is not None
#
#     # def begin(self):
#     #     pass
#
#     def before_run(self, run_context):
#         self.prof.set_operation(self.operation)
#         self.is_running = True
#
#     def after_run(self, run_context, run_values):
#         self.prof.end_operation(self.operation)
#         self.is_running = False
#
#     def end(self, session):
#         """
#         If sess.run()) raises OutOfRangeError or StopIteration,
#         after_run will not get called.
#
#         Nevertheless, make sure to stop profiling.
#
#         NOTE: this won't get called if other exceptions occur though.
#
#         Q: Why would OutOfRangeError or StopIteration be raised?
#         """
#         if self.is_running:
#             self.prof.end_operation(self.operation)
#             self.is_running = False

class BeforeRunSetOpEstimatorHook(tf.train.SessionRunHook):
    def __init__(self, operation, prof=None):
        """
        Hook for Estimator callback API to add profiling set_operation/end_operation
        annotations before/after invoking TensorFlow session.run(...) calls.

        :param operation:
            The name of the operation to use for prof.set_operation(...).
        """
        self.operation = operation
        if prof is None:
            self.prof = iml_profiler.api.prof
        else:
            self.prof = prof
        assert self.prof is not None

    def before_run(self, run_context):
        self.prof.set_operation(self.operation)

class AfterRunEndOpEstimatorHook(tf.train.SessionRunHook):
    def __init__(self, operation, prof=None):
        """
        Hook for Estimator callback API to add profiling set_operation/end_operation
        annotations before/after invoking TensorFlow session.run(...) calls.

        :param operation:
            The name of the operation to use for prof.set_operation(...).
        """
        self.operation = operation
        if prof is None:
            self.prof = iml_profiler.api.prof
        else:
            self.prof = prof
        assert self.prof is not None

    def after_run(self, run_context, run_values):
        self.prof.end_operation(self.operation)

# class SaveModelEstimatorHook(CheckpointSaverListener):
#     def __init__(self, prof=None):
#         self.operation = 'estimator_save_model'
#         if prof is None:
#             self.prof = iml_profiler.api.prof
#         else:
#             self.prof = prof
#         assert self.prof is not None
#
#     def before_save(self, session, global_step_value):
#         self.prof.set_operation(self.operation)
#
#     def after_save(self, session, global_step_value):
#         self.prof.end_operation(self.operation)

class BeforeSaveSetOpEstimatorHook(CheckpointSaverListener):
    def __init__(self, operation, prof=None):
        self.operation = operation
        if prof is None:
            self.prof = iml_profiler.api.prof
        else:
            self.prof = prof
        assert self.prof is not None

    def before_save(self, session, global_step_value):
        self.prof.set_operation(self.operation)

class AfterSaveEndOpEstimatorHook(CheckpointSaverListener):
    def __init__(self, operation, prof=None):
        self.operation = operation
        if prof is None:
            self.prof = iml_profiler.api.prof
        else:
            self.prof = prof
        assert self.prof is not None

    def after_save(self, session, global_step_value):
        self.prof.end_operation(self.operation)

class SetOpTrainEstimatorHook(BeforeRunSetOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_train', prof=prof)
class EndOpTrainEstimatorHook(AfterRunEndOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_train', prof=prof)

class SetOpPredictEstimatorHook(BeforeRunSetOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_predict', prof=prof)
class EndOpPredictEstimatorHook(AfterRunEndOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_predict', prof=prof)

class SetOpEvaluateEstimatorHook(BeforeRunSetOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_evaluate', prof=prof)
class EndOpEvaluateEstimatorHook(AfterRunEndOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_evaluate', prof=prof)

class SetOpSaveEstimatorHook(BeforeSaveSetOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_save_model', prof=prof)
class EndOpSaveEstimatorHook(AfterSaveEndOpEstimatorHook):
    def __init__(self, prof=None):
        super().__init__(operation='estimator_save_model', prof=prof)

# class PredictEstimatorHook(ProfileEstimatorHook):
#     def __init__(self, prof=None):
#         super().__init__(operation='estimator_predict', prof=prof)
# class EvaluateEstimatorHook(ProfileEstimatorHook):
#     def __init__(self, prof=None):
#         super().__init__(operation='estimator_evaluate', prof=prof)
