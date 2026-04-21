import warnings


def _msg_yellow(message: str) -> None:
    """Print an informational message in yellow without a 'Warning:' prefix."""
    print(f"\033[93m{message}\033[0m")
    print()


def _warn_yellow(message: str) -> None:
    """Print a short yellow warning message."""
    original_warning_formatter = warnings.formatwarning

    def _warning_format(msg, category, filename, lineno, line=None):
        _ = category, filename, lineno, line
        return f"\033[93mWarning: {msg}\033[0m\n"

    warnings.formatwarning = _warning_format
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(message, UserWarning, stacklevel=2)
    finally:
        warnings.formatwarning = original_warning_formatter
