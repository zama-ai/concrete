#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt for license information.

"""Wrapper for native Cpp objects."""


class WrapperCpp:
    """Wrapper base class for native Cpp objects.

    Initialization should mainly store the wrapped object, and future calls to the wrapper will be forwarded
    to it. A static wrap method is provided to be more explicit. Wrappers should always be constructed using
    the new method, which construct the Cpp object using the provided arguments, then wrap it. Classes that
    inherit from this class should preferably type check the wrapped object during calls to init, and
    reimplement the new method if the class is meant to be constructed.
    """

    def __init__(self, cpp_obj):
        self._cpp_obj = cpp_obj

    @classmethod
    def wrap(cls, cpp_obj) -> "WrapperCpp":
        """Wrap the Cpp object into a Python object.

        Args:
            cpp_obj: object to wrap

        Returns:
            WrapperCpp: wrapper
        """
        return cls(cpp_obj)

    @staticmethod
    def new(*args, **kwargs):
        """Create a new wrapper by building the underlying object with a specific set of arguments."""
        raise RuntimeError("This class shouldn't be built")

    def cpp(self):
        """Return the Cpp wrapped object."""
        return self._cpp_obj
