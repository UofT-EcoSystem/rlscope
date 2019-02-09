"""
profiler.session.ACTIVE_SESSIONS keeps track of "currently active sessions".

These are the sessions we need to check for trace-data during the DUMP phase of profiling.

See SessionWrapper for details.
"""
import tensorflow as tf
import threading
import pprint

from profiler import tensorflow_profile_context

DEBUG = tensorflow_profile_context.DEBUG

"""
Wrap tf.Session / tf.InteractiveSession.
"""
SETUP_DONE = False
def setup():
    global SETUP_DONE
    assert not SETUP_DONE

    setup_wrap_BaseSession()

    # setup_wrap_Session()
    # setup_wrap_InteractiveSession()

    SETUP_DONE = True

from tensorflow.python.client.session import BaseSession
_original_BaseSession_init = None
def setup_wrap_BaseSession():
    global _original_BaseSession_init
    global _original_BaseSession_close

    _original_BaseSession_init = BaseSession.__init__
    _original_BaseSession_close = BaseSession.close

    BaseSession.__init__ = _wrapped_BaseSession_init
    BaseSession.close = _wrapped_BaseSession_close

# old_Session = None
# def setup_wrap_Session():
#     global old_Session
#     old_Session = tf.Session
#     tf.Session = SessionWrapper
#
# old_InteractiveSession = None
# def setup_wrap_InteractiveSession():
#     global old_InteractiveSession
#     old_InteractiveSession = tf.InteractiveSession
#     tf.InteractiveSession = InteractiveSessionWrapper

# TODO:
# We also need to wrap tf.InteractiveSession.

class SessionActiveHook:
    """
    Interface for adding hooks for when a session becomes active.
    """
    def __init__(self):
        pass

    def before_active(self, session):
        pass

    def after_active(self, session):
        pass

class SessionInactiveHook:
    """
    Interface for adding hooks for when a session becomes inactive.
    """
    def __init__(self):
        pass

    def before_inactive(self, session):
        pass

    def after_inactive(self, session):
        pass

ACTIVE_HOOKS = []
def register_session_active_hook(hook : SessionActiveHook):
    ACTIVE_HOOKS.append(hook)
def unregister_session_active_hook(hook : SessionActiveHook):
    ACTIVE_HOOKS.remove(hook)

INACTIVE_HOOKS = []
def register_session_inactive_hook(hook : SessionInactiveHook):
    INACTIVE_HOOKS.append(hook)
def unregister_session_inactive_hook(hook : SessionInactiveHook):
    INACTIVE_HOOKS.remove(hook)

# NOTE: would be nice to have an INACTIVE session list, where once we dump the session, we no longer need to keep track of it anymore

# All currently active tf.Session objects.
#
# Q: Is there an internal variable in TensorFlow that already tracks this for us?
# A: No, doesn't look like it.
# ACTIVE_SESSIONS = []
ACTIVE_SESSIONS = set()

def _make_inactive(session):
    global ACTIVE_SESSIONS
    if session in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS.remove(session)
def _make_active(session):
    global ACTIVE_SESSIONS, THREAD_IDS
    # ACTIVE_SESSIONS.append(session)
    assert session not in ACTIVE_SESSIONS
    ACTIVE_SESSIONS.add(session)

def _before_active_hooks(session):
    global ACTIVE_HOOKS
    for hook in ACTIVE_HOOKS:
        hook.before_active(session)

def _after_active_hooks(session):
    global ACTIVE_HOOKS
    for hook in ACTIVE_HOOKS:
        hook.after_active(session)

def _before_inactive_hooks(session):
    global INACTIVE_HOOKS
    for hook in INACTIVE_HOOKS:
        hook.before_inactive(session)

def _after_inactive_hooks(session):
    global INACTIVE_HOOKS
    for hook in INACTIVE_HOOKS:
        hook.after_inactive(session)

# Autoincremented 
_NEXT_SESSION_ID = 0
def _get_next_session_id():
    global _NEXT_SESSION_ID
    session_id = _NEXT_SESSION_ID
    if DEBUG:
        print("> DEBUG: session.py: alloc session_id = {id}".format(id=session_id))
    _NEXT_SESSION_ID += 1
    return session_id

def _wrapped_BaseSession_init(self, *args, **kwargs):
    # print("> Wrapped: {name}".format(
    #     name=_wrapped_BaseSession_init.__name__,
    # ))
    self.session_id = _get_next_session_id()
    _before_active_hooks(session=self)
    _make_active(session=self)
    _original_BaseSession_init(self, *args, **kwargs)
    _after_active_hooks(session=self)

def _wrapped_BaseSession_close(self):
    """
    A session is inactive if:
    1. The Session has been closed.
    """
    # print("> Wrapped: {name}".format(
    #     name=_wrapped_BaseSession_close.__name__,
    # ))
    _before_inactive_hooks(session=self)
    _make_inactive(session=self)
    _original_BaseSession_close(self)
    _after_inactive_hooks(session=self)


# NOTE: This approach does NOT work for tf.estimator.Estimator.train(...).
# In particular, it must use some internal Session class that inherits from BaseSession,
# but isn't one of tf.Session or tf.InteractiveSession.
#
# So instead, we just hack around it by manually wrapping BaseSession.__init__/BaseSession.close.
# (above).

# class SessionWrapper(tf.Session):
#     """
#     Wrapper that keeps track of what the "currently active sessions" are.
#
#     A session is inactive if:
#     1. The Session has been closed.
#     # 2. The Session has has gone out of scope
#     #    (we detect this when __del__ has been called).
#
#     PROBLEM: We cannot know when a session goes out of scope without being closed...
#     Yes we can, override __del__.
#     PROBLEM: It's a bad idea to keep a Session after __del__ since its C++-side
#     Session gets deleted.
#
#     PROBLEM/SOLUTION/COMPROMISE: Don't override __del__; if the user never calls
#     Session.close(), that's their fault. We'll just hold onto the Session forever.
#     OH WELL.
#     """
#
#     def __init__(self, *args, **kwargs):
#         self.session_id = _get_next_session_id()
#         _before_active_hooks(session=self)
#         _make_active(session=self)
#         super().__init__(*args, **kwargs)
#         _after_active_hooks(session=self)
#
#     # def __del__(self):
#     #     """
#     #     A session is inactive if:
#     #     2. The Session has has gone out of scope
#     #        (we detect this when __del__ has been called).
#     #     """
#     #     _before_inactive_hooks(session=self)
#     #     _make_inactive(session=self)
#     #     super().__del__()
#     #     _after_inactive_hooks(session=self)
#
#     def close(self):
#         """
#         A session is inactive if:
#         1. The Session has been closed.
#         """
#         _before_inactive_hooks(session=self)
#         _make_inactive(session=self)
#         super().close()
#         _after_inactive_hooks(session=self)
#
# class InteractiveSessionWrapper(tf.InteractiveSession):
#     """
#     Same as SessionWrapper.
#     """
#     def __init__(self, *args, **kwargs):
#         self.session_id = _get_next_session_id()
#         _before_active_hooks(session=self)
#         _make_active(session=self)
#         super().__init__(*args, **kwargs)
#         _after_active_hooks(session=self)
#
#     # def __del__(self):
#     #     _before_inactive_hooks(session=self)
#     #     _make_inactive(session=self)
#     #     super().__del__()
#     #     _after_inactive_hooks(session=self)
#
#     def close(self):
#         _before_inactive_hooks(session=self)
#         _make_inactive(session=self)
#         super().close()
#         _after_inactive_hooks(session=self)
