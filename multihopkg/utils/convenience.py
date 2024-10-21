from typing import Type, TypeVar, Any

# Define a generic type variable
T = TypeVar("T")


# TOREM: When we finish the first stage of debugging
def create_placeholder(expected_type: Type[T], name: str, location: str) -> Any:
    """Creates a placeholder function that raises NotImplementedError.
    Args:
        expected_type: The expected return type of the function.
    Returns:
        A function that raises NotImplementedError.
    """

    def placeholder(*args, **kwargs) -> T:
        raise NotImplementedError(
            f"{name}, at {location} is a placeholder and is expected to return {expected_type.__name__}.\n"
            "If you see this error it means you commited to changing this later"
        )

    return placeholder
