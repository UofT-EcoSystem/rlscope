"""
RL-Scope related errors and exceptions.
"""

class RLScopeConfigurationError(Exception):
    """
    Error raised when the host/container isn't properly configured.
    For example:
    - installation dependency missing
    """
    pass

class RLScopeAPIError(Exception):
    """
    Error raised when the rlscope user API is used improperly.
    """
    pass

class RLScopeRunError(Exception):
    """
    Error raised when an error is encountered while running the training script and collecting trace files.
    """
    pass

class RLScopeAnalysisError(Exception):
    """
    Error raised when an error is encountered while processing trace files.
    """
    pass
