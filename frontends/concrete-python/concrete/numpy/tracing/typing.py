"""
Declaration of type annotation.
"""

from typing import Any

from ..dtypes import Float, SignedInteger, UnsignedInteger
from ..values import Value
from .tracer import ScalarAnnotation, TensorAnnotation

# pylint: disable=function-redefined,invalid-name,no-self-use,too-many-lines,using-constant-test
# ruff: noqa


# We'll pull a little trick on mypy
# Basically, this branch is never executed during runtime
# But, mypy will use the information within anyway
# So, it'll think our types are `Any` and it'll stop complaining when used with numpy

if False:

    f32 = Any
    f64 = Any

    int1 = Any
    int2 = Any
    int3 = Any
    int4 = Any
    int5 = Any
    int6 = Any
    int7 = Any
    int8 = Any
    int9 = Any
    int10 = Any
    int11 = Any
    int12 = Any
    int13 = Any
    int14 = Any
    int15 = Any
    int16 = Any
    int17 = Any
    int18 = Any
    int19 = Any
    int20 = Any
    int21 = Any
    int22 = Any
    int23 = Any
    int24 = Any
    int25 = Any
    int26 = Any
    int27 = Any
    int28 = Any
    int29 = Any
    int30 = Any
    int31 = Any
    int32 = Any
    int33 = Any
    int34 = Any
    int35 = Any
    int36 = Any
    int37 = Any
    int38 = Any
    int39 = Any
    int40 = Any
    int41 = Any
    int42 = Any
    int43 = Any
    int44 = Any
    int45 = Any
    int46 = Any
    int47 = Any
    int48 = Any
    int49 = Any
    int50 = Any
    int51 = Any
    int52 = Any
    int53 = Any
    int54 = Any
    int55 = Any
    int56 = Any
    int57 = Any
    int58 = Any
    int59 = Any
    int60 = Any
    int61 = Any
    int62 = Any
    int63 = Any
    int64 = Any

    uint1 = Any
    uint2 = Any
    uint3 = Any
    uint4 = Any
    uint5 = Any
    uint6 = Any
    uint7 = Any
    uint8 = Any
    uint9 = Any
    uint10 = Any
    uint11 = Any
    uint12 = Any
    uint13 = Any
    uint14 = Any
    uint15 = Any
    uint16 = Any
    uint17 = Any
    uint18 = Any
    uint19 = Any
    uint20 = Any
    uint21 = Any
    uint22 = Any
    uint23 = Any
    uint24 = Any
    uint25 = Any
    uint26 = Any
    uint27 = Any
    uint28 = Any
    uint29 = Any
    uint30 = Any
    uint31 = Any
    uint32 = Any
    uint33 = Any
    uint34 = Any
    uint35 = Any
    uint36 = Any
    uint37 = Any
    uint38 = Any
    uint39 = Any
    uint40 = Any
    uint41 = Any
    uint42 = Any
    uint43 = Any
    uint44 = Any
    uint45 = Any
    uint46 = Any
    uint47 = Any
    uint48 = Any
    uint49 = Any
    uint50 = Any
    uint51 = Any
    uint52 = Any
    uint53 = Any
    uint54 = Any
    uint55 = Any
    uint56 = Any
    uint57 = Any
    uint58 = Any
    uint59 = Any
    uint60 = Any
    uint61 = Any
    uint62 = Any
    uint63 = Any
    uint64 = Any
    tensor = Any


class f32(ScalarAnnotation):  # type: ignore
    """
    Scalar f32 annotation.
    """

    dtype = Float(32)


class f64(ScalarAnnotation):  # type: ignore
    """
    Scalar f64 annotation.
    """

    dtype = Float(64)


class int1(ScalarAnnotation):  # type: ignore
    """
    Scalar int1 annotation.
    """

    dtype = SignedInteger(1)


class int2(ScalarAnnotation):  # type: ignore
    """
    Scalar int2 annotation.
    """

    dtype = SignedInteger(2)


class int3(ScalarAnnotation):  # type: ignore
    """
    Scalar int3 annotation.
    """

    dtype = SignedInteger(3)


class int4(ScalarAnnotation):  # type: ignore
    """
    Scalar int4 annotation.
    """

    dtype = SignedInteger(4)


class int5(ScalarAnnotation):  # type: ignore
    """
    Scalar int5 annotation.
    """

    dtype = SignedInteger(5)


class int6(ScalarAnnotation):  # type: ignore
    """
    Scalar int6 annotation.
    """

    dtype = SignedInteger(6)


class int7(ScalarAnnotation):  # type: ignore
    """
    Scalar int7 annotation.
    """

    dtype = SignedInteger(7)


class int8(ScalarAnnotation):  # type: ignore
    """
    Scalar int8 annotation.
    """

    dtype = SignedInteger(8)


class int9(ScalarAnnotation):  # type: ignore
    """
    Scalar int9 annotation.
    """

    dtype = SignedInteger(9)


class int10(ScalarAnnotation):  # type: ignore
    """
    Scalar int10 annotation.
    """

    dtype = SignedInteger(10)


class int11(ScalarAnnotation):  # type: ignore
    """
    Scalar int11 annotation.
    """

    dtype = SignedInteger(11)


class int12(ScalarAnnotation):  # type: ignore
    """
    Scalar int12 annotation.
    """

    dtype = SignedInteger(12)


class int13(ScalarAnnotation):  # type: ignore
    """
    Scalar int13 annotation.
    """

    dtype = SignedInteger(13)


class int14(ScalarAnnotation):  # type: ignore
    """
    Scalar int14 annotation.
    """

    dtype = SignedInteger(14)


class int15(ScalarAnnotation):  # type: ignore
    """
    Scalar int15 annotation.
    """

    dtype = SignedInteger(15)


class int16(ScalarAnnotation):  # type: ignore
    """
    Scalar int16 annotation.
    """

    dtype = SignedInteger(16)


class int17(ScalarAnnotation):  # type: ignore
    """
    Scalar int17 annotation.
    """

    dtype = SignedInteger(17)


class int18(ScalarAnnotation):  # type: ignore
    """
    Scalar int18 annotation.
    """

    dtype = SignedInteger(18)


class int19(ScalarAnnotation):  # type: ignore
    """
    Scalar int19 annotation.
    """

    dtype = SignedInteger(19)


class int20(ScalarAnnotation):  # type: ignore
    """
    Scalar int20 annotation.
    """

    dtype = SignedInteger(20)


class int21(ScalarAnnotation):  # type: ignore
    """
    Scalar int21 annotation.
    """

    dtype = SignedInteger(21)


class int22(ScalarAnnotation):  # type: ignore
    """
    Scalar int22 annotation.
    """

    dtype = SignedInteger(22)


class int23(ScalarAnnotation):  # type: ignore
    """
    Scalar int23 annotation.
    """

    dtype = SignedInteger(23)


class int24(ScalarAnnotation):  # type: ignore
    """
    Scalar int24 annotation.
    """

    dtype = SignedInteger(24)


class int25(ScalarAnnotation):  # type: ignore
    """
    Scalar int25 annotation.
    """

    dtype = SignedInteger(25)


class int26(ScalarAnnotation):  # type: ignore
    """
    Scalar int26 annotation.
    """

    dtype = SignedInteger(26)


class int27(ScalarAnnotation):  # type: ignore
    """
    Scalar int27 annotation.
    """

    dtype = SignedInteger(27)


class int28(ScalarAnnotation):  # type: ignore
    """
    Scalar int28 annotation.
    """

    dtype = SignedInteger(28)


class int29(ScalarAnnotation):  # type: ignore
    """
    Scalar int29 annotation.
    """

    dtype = SignedInteger(29)


class int30(ScalarAnnotation):  # type: ignore
    """
    Scalar int30 annotation.
    """

    dtype = SignedInteger(30)


class int31(ScalarAnnotation):  # type: ignore
    """
    Scalar int31 annotation.
    """

    dtype = SignedInteger(31)


class int32(ScalarAnnotation):  # type: ignore
    """
    Scalar int32 annotation.
    """

    dtype = SignedInteger(32)


class int33(ScalarAnnotation):  # type: ignore
    """
    Scalar int33 annotation.
    """

    dtype = SignedInteger(33)


class int34(ScalarAnnotation):  # type: ignore
    """
    Scalar int34 annotation.
    """

    dtype = SignedInteger(34)


class int35(ScalarAnnotation):  # type: ignore
    """
    Scalar int35 annotation.
    """

    dtype = SignedInteger(35)


class int36(ScalarAnnotation):  # type: ignore
    """
    Scalar int36 annotation.
    """

    dtype = SignedInteger(36)


class int37(ScalarAnnotation):  # type: ignore
    """
    Scalar int37 annotation.
    """

    dtype = SignedInteger(37)


class int38(ScalarAnnotation):  # type: ignore
    """
    Scalar int38 annotation.
    """

    dtype = SignedInteger(38)


class int39(ScalarAnnotation):  # type: ignore
    """
    Scalar int39 annotation.
    """

    dtype = SignedInteger(39)


class int40(ScalarAnnotation):  # type: ignore
    """
    Scalar int40 annotation.
    """

    dtype = SignedInteger(40)


class int41(ScalarAnnotation):  # type: ignore
    """
    Scalar int41 annotation.
    """

    dtype = SignedInteger(41)


class int42(ScalarAnnotation):  # type: ignore
    """
    Scalar int42 annotation.
    """

    dtype = SignedInteger(42)


class int43(ScalarAnnotation):  # type: ignore
    """
    Scalar int43 annotation.
    """

    dtype = SignedInteger(43)


class int44(ScalarAnnotation):  # type: ignore
    """
    Scalar int44 annotation.
    """

    dtype = SignedInteger(44)


class int45(ScalarAnnotation):  # type: ignore
    """
    Scalar int45 annotation.
    """

    dtype = SignedInteger(45)


class int46(ScalarAnnotation):  # type: ignore
    """
    Scalar int46 annotation.
    """

    dtype = SignedInteger(46)


class int47(ScalarAnnotation):  # type: ignore
    """
    Scalar int47 annotation.
    """

    dtype = SignedInteger(47)


class int48(ScalarAnnotation):  # type: ignore
    """
    Scalar int48 annotation.
    """

    dtype = SignedInteger(48)


class int49(ScalarAnnotation):  # type: ignore
    """
    Scalar int49 annotation.
    """

    dtype = SignedInteger(49)


class int50(ScalarAnnotation):  # type: ignore
    """
    Scalar int50 annotation.
    """

    dtype = SignedInteger(50)


class int51(ScalarAnnotation):  # type: ignore
    """
    Scalar int51 annotation.
    """

    dtype = SignedInteger(51)


class int52(ScalarAnnotation):  # type: ignore
    """
    Scalar int52 annotation.
    """

    dtype = SignedInteger(52)


class int53(ScalarAnnotation):  # type: ignore
    """
    Scalar int53 annotation.
    """

    dtype = SignedInteger(53)


class int54(ScalarAnnotation):  # type: ignore
    """
    Scalar int54 annotation.
    """

    dtype = SignedInteger(54)


class int55(ScalarAnnotation):  # type: ignore
    """
    Scalar int55 annotation.
    """

    dtype = SignedInteger(55)


class int56(ScalarAnnotation):  # type: ignore
    """
    Scalar int56 annotation.
    """

    dtype = SignedInteger(56)


class int57(ScalarAnnotation):  # type: ignore
    """
    Scalar int57 annotation.
    """

    dtype = SignedInteger(57)


class int58(ScalarAnnotation):  # type: ignore
    """
    Scalar int58 annotation.
    """

    dtype = SignedInteger(58)


class int59(ScalarAnnotation):  # type: ignore
    """
    Scalar int59 annotation.
    """

    dtype = SignedInteger(59)


class int60(ScalarAnnotation):  # type: ignore
    """
    Scalar int60 annotation.
    """

    dtype = SignedInteger(60)


class int61(ScalarAnnotation):  # type: ignore
    """
    Scalar int61 annotation.
    """

    dtype = SignedInteger(61)


class int62(ScalarAnnotation):  # type: ignore
    """
    Scalar int62 annotation.
    """

    dtype = SignedInteger(62)


class int63(ScalarAnnotation):  # type: ignore
    """
    Scalar int63 annotation.
    """

    dtype = SignedInteger(63)


class int64(ScalarAnnotation):  # type: ignore
    """
    Scalar int64 annotation.
    """

    dtype = SignedInteger(64)


class uint1(ScalarAnnotation):  # type: ignore
    """
    Scalar uint1 annotation.
    """

    dtype = UnsignedInteger(1)


class uint2(ScalarAnnotation):  # type: ignore
    """
    Scalar uint2 annotation.
    """

    dtype = UnsignedInteger(2)


class uint3(ScalarAnnotation):  # type: ignore
    """
    Scalar uint3 annotation.
    """

    dtype = UnsignedInteger(3)


class uint4(ScalarAnnotation):  # type: ignore
    """
    Scalar uint4 annotation.
    """

    dtype = UnsignedInteger(4)


class uint5(ScalarAnnotation):  # type: ignore
    """
    Scalar uint5 annotation.
    """

    dtype = UnsignedInteger(5)


class uint6(ScalarAnnotation):  # type: ignore
    """
    Scalar uint6 annotation.
    """

    dtype = UnsignedInteger(6)


class uint7(ScalarAnnotation):  # type: ignore
    """
    Scalar uint7 annotation.
    """

    dtype = UnsignedInteger(7)


class uint8(ScalarAnnotation):  # type: ignore
    """
    Scalar uint8 annotation.
    """

    dtype = UnsignedInteger(8)


class uint9(ScalarAnnotation):  # type: ignore
    """
    Scalar uint9 annotation.
    """

    dtype = UnsignedInteger(9)


class uint10(ScalarAnnotation):  # type: ignore
    """
    Scalar uint10 annotation.
    """

    dtype = UnsignedInteger(10)


class uint11(ScalarAnnotation):  # type: ignore
    """
    Scalar uint11 annotation.
    """

    dtype = UnsignedInteger(11)


class uint12(ScalarAnnotation):  # type: ignore
    """
    Scalar uint12 annotation.
    """

    dtype = UnsignedInteger(12)


class uint13(ScalarAnnotation):  # type: ignore
    """
    Scalar uint13 annotation.
    """

    dtype = UnsignedInteger(13)


class uint14(ScalarAnnotation):  # type: ignore
    """
    Scalar uint14 annotation.
    """

    dtype = UnsignedInteger(14)


class uint15(ScalarAnnotation):  # type: ignore
    """
    Scalar uint15 annotation.
    """

    dtype = UnsignedInteger(15)


class uint16(ScalarAnnotation):  # type: ignore
    """
    Scalar uint16 annotation.
    """

    dtype = UnsignedInteger(16)


class uint17(ScalarAnnotation):  # type: ignore
    """
    Scalar uint17 annotation.
    """

    dtype = UnsignedInteger(17)


class uint18(ScalarAnnotation):  # type: ignore
    """
    Scalar uint18 annotation.
    """

    dtype = UnsignedInteger(18)


class uint19(ScalarAnnotation):  # type: ignore
    """
    Scalar uint19 annotation.
    """

    dtype = UnsignedInteger(19)


class uint20(ScalarAnnotation):  # type: ignore
    """
    Scalar uint20 annotation.
    """

    dtype = UnsignedInteger(20)


class uint21(ScalarAnnotation):  # type: ignore
    """
    Scalar uint21 annotation.
    """

    dtype = UnsignedInteger(21)


class uint22(ScalarAnnotation):  # type: ignore
    """
    Scalar uint22 annotation.
    """

    dtype = UnsignedInteger(22)


class uint23(ScalarAnnotation):  # type: ignore
    """
    Scalar uint23 annotation.
    """

    dtype = UnsignedInteger(23)


class uint24(ScalarAnnotation):  # type: ignore
    """
    Scalar uint24 annotation.
    """

    dtype = UnsignedInteger(24)


class uint25(ScalarAnnotation):  # type: ignore
    """
    Scalar uint25 annotation.
    """

    dtype = UnsignedInteger(25)


class uint26(ScalarAnnotation):  # type: ignore
    """
    Scalar uint26 annotation.
    """

    dtype = UnsignedInteger(26)


class uint27(ScalarAnnotation):  # type: ignore
    """
    Scalar uint27 annotation.
    """

    dtype = UnsignedInteger(27)


class uint28(ScalarAnnotation):  # type: ignore
    """
    Scalar uint28 annotation.
    """

    dtype = UnsignedInteger(28)


class uint29(ScalarAnnotation):  # type: ignore
    """
    Scalar uint29 annotation.
    """

    dtype = UnsignedInteger(29)


class uint30(ScalarAnnotation):  # type: ignore
    """
    Scalar uint30 annotation.
    """

    dtype = UnsignedInteger(30)


class uint31(ScalarAnnotation):  # type: ignore
    """
    Scalar uint31 annotation.
    """

    dtype = UnsignedInteger(31)


class uint32(ScalarAnnotation):  # type: ignore
    """
    Scalar uint32 annotation.
    """

    dtype = UnsignedInteger(32)


class uint33(ScalarAnnotation):  # type: ignore
    """
    Scalar uint33 annotation.
    """

    dtype = UnsignedInteger(33)


class uint34(ScalarAnnotation):  # type: ignore
    """
    Scalar uint34 annotation.
    """

    dtype = UnsignedInteger(34)


class uint35(ScalarAnnotation):  # type: ignore
    """
    Scalar uint35 annotation.
    """

    dtype = UnsignedInteger(35)


class uint36(ScalarAnnotation):  # type: ignore
    """
    Scalar uint36 annotation.
    """

    dtype = UnsignedInteger(36)


class uint37(ScalarAnnotation):  # type: ignore
    """
    Scalar uint37 annotation.
    """

    dtype = UnsignedInteger(37)


class uint38(ScalarAnnotation):  # type: ignore
    """
    Scalar uint38 annotation.
    """

    dtype = UnsignedInteger(38)


class uint39(ScalarAnnotation):  # type: ignore
    """
    Scalar uint39 annotation.
    """

    dtype = UnsignedInteger(39)


class uint40(ScalarAnnotation):  # type: ignore
    """
    Scalar uint40 annotation.
    """

    dtype = UnsignedInteger(40)


class uint41(ScalarAnnotation):  # type: ignore
    """
    Scalar uint41 annotation.
    """

    dtype = UnsignedInteger(41)


class uint42(ScalarAnnotation):  # type: ignore
    """
    Scalar uint42 annotation.
    """

    dtype = UnsignedInteger(42)


class uint43(ScalarAnnotation):  # type: ignore
    """
    Scalar uint43 annotation.
    """

    dtype = UnsignedInteger(43)


class uint44(ScalarAnnotation):  # type: ignore
    """
    Scalar uint44 annotation.
    """

    dtype = UnsignedInteger(44)


class uint45(ScalarAnnotation):  # type: ignore
    """
    Scalar uint45 annotation.
    """

    dtype = UnsignedInteger(45)


class uint46(ScalarAnnotation):  # type: ignore
    """
    Scalar uint46 annotation.
    """

    dtype = UnsignedInteger(46)


class uint47(ScalarAnnotation):  # type: ignore
    """
    Scalar uint47 annotation.
    """

    dtype = UnsignedInteger(47)


class uint48(ScalarAnnotation):  # type: ignore
    """
    Scalar uint48 annotation.
    """

    dtype = UnsignedInteger(48)


class uint49(ScalarAnnotation):  # type: ignore
    """
    Scalar uint49 annotation.
    """

    dtype = UnsignedInteger(49)


class uint50(ScalarAnnotation):  # type: ignore
    """
    Scalar uint50 annotation.
    """

    dtype = UnsignedInteger(50)


class uint51(ScalarAnnotation):  # type: ignore
    """
    Scalar uint51 annotation.
    """

    dtype = UnsignedInteger(51)


class uint52(ScalarAnnotation):  # type: ignore
    """
    Scalar uint52 annotation.
    """

    dtype = UnsignedInteger(52)


class uint53(ScalarAnnotation):  # type: ignore
    """
    Scalar uint53 annotation.
    """

    dtype = UnsignedInteger(53)


class uint54(ScalarAnnotation):  # type: ignore
    """
    Scalar uint54 annotation.
    """

    dtype = UnsignedInteger(54)


class uint55(ScalarAnnotation):  # type: ignore
    """
    Scalar uint55 annotation.
    """

    dtype = UnsignedInteger(55)


class uint56(ScalarAnnotation):  # type: ignore
    """
    Scalar uint56 annotation.
    """

    dtype = UnsignedInteger(56)


class uint57(ScalarAnnotation):  # type: ignore
    """
    Scalar uint57 annotation.
    """

    dtype = UnsignedInteger(57)


class uint58(ScalarAnnotation):  # type: ignore
    """
    Scalar uint58 annotation.
    """

    dtype = UnsignedInteger(58)


class uint59(ScalarAnnotation):  # type: ignore
    """
    Scalar uint59 annotation.
    """

    dtype = UnsignedInteger(59)


class uint60(ScalarAnnotation):  # type: ignore
    """
    Scalar uint60 annotation.
    """

    dtype = UnsignedInteger(60)


class uint61(ScalarAnnotation):  # type: ignore
    """
    Scalar uint61 annotation.
    """

    dtype = UnsignedInteger(61)


class uint62(ScalarAnnotation):  # type: ignore
    """
    Scalar uint62 annotation.
    """

    dtype = UnsignedInteger(62)


class uint63(ScalarAnnotation):  # type: ignore
    """
    Scalar uint63 annotation.
    """

    dtype = UnsignedInteger(63)


class uint64(ScalarAnnotation):  # type: ignore
    """
    Scalar uint64 annotation.
    """

    dtype = UnsignedInteger(64)


class tensor(TensorAnnotation):  # type: ignore
    """
    Tensor annotation.
    """

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple):
            item = (item,)

        annotation = item[0]
        if not issubclass(annotation, ScalarAnnotation):
            raise ValueError(
                f"First argument to tensor annotations should be a "
                f"concrete-numpy data type (e.g., cnp.uint4) "
                f"not {annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)}"
            )

        if len(item) == 1:
            raise ValueError(
                "Tensor annotations should have a shape (e.g., cnp.tensor[cnp.uint4, 3, 2])"
            )

        shape = item[1:]
        if not all(isinstance(x, int) for x in shape):
            raise ValueError("Tensor annotation shape elements must be 'int'")

        return Value(dtype=annotation.dtype, shape=shape, is_encrypted=False)
