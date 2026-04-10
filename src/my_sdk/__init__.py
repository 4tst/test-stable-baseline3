def add(a: int, b: int) -> int:
    """
    add two integers

    Args:
        a (int): first integer
        b (int): second integer

    Returns:
        int: sum of a and b
    """
    return a + b


try:
    from .version import __version__
except ImportError as e:
    print("version module not found, please create `my_sdk/version.py` first")
    raise e
