"""Common python helpers."""

from typing import Any, Callable, Iterable, Mapping, Tuple, Union


def update_and_return_dict(
    dict_to_update: dict, update_values: Union[Mapping, Iterable[Tuple[Any, Any]]]
) -> dict:
    """Update a dictionary and return the ref to the dictionary that was updated.

    Args:
        dict_to_update (dict): the dict to update
        update_values (Union[Mapping, Iterable[Tuple[Any, Any]]]): the values to update the dict
            with

    Returns:
        dict: the dict that was just updated.
    """
    dict_to_update.update(update_values)
    return dict_to_update


def catch(func: Callable, *args, **kwargs) -> Union[Any, None]:
    """Execute func by passing args and kwargs. Catch exceptions and return None in case of failure.

    Args:
        func (Callable): function to execute and catch exceptions from

    Returns:
        Union[Any, None]: the function result if there was no exception, None otherwise.
    """
    try:
        return func(*args, **kwargs)
    except Exception:  # pylint: disable=broad-except
        return None
