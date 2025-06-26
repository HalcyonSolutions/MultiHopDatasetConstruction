import functools

def deprecated(deprecation_date, reason):
    """
    A decorator to mark functions as deprecated.
    Parameters:
    - deprecation_date (str): The date when the function was deprecated.
    - reason (str): The reason why the function is deprecated.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(
                f"The function '{func.__name__}' is deprecated as of {deprecation_date}. "
                f"Reason: {reason}"
            )
        return wrapper
    return decorator
# Example usage:
