"""User-facing errors.

Anything raised as a BenchVisionError is expected to be shown verbatim to the
person at the bench (via the MCP client), so messages must say what went wrong
and what to do about it, never just echo an internal exception.
"""


class BenchVisionError(Exception):
    """An anticipated failure with a message fit for the user."""


class ConfigError(BenchVisionError):
    """config.toml is missing, unreadable, or invalid."""


class CameraError(BenchVisionError):
    """A camera could not be opened or read."""


class CameraOpenError(CameraError):
    """The camera is missing or could not be opened (as opposed to failing after it opened)."""


class CameraMissingError(CameraOpenError):
    """The camera's device isn't there at all (unplugged, wrong by-id path, mock image missing)."""
