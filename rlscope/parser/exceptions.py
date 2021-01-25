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

