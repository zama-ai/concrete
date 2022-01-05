# pylint: disable=too-many-lines
import random

import numpy as np
import progress

import concrete.numpy as hnp


@progress.track(
    [
        # Addition
        {
            "id": "x-plus-42-scalar",
            "name": "x + 42 {Scalar}",
            "parameters": {
                "function": lambda x: x + 42,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 85,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-42-tensor-2x3",
            "name": "x + 42 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x + 42,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 85,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-10-20-30-tensor-3",
            "name": "x + [10, 20, 30] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x + np.array([10, 20, 30], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 97,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-10-20-30-tensor-2x3",
            "name": "x + [10, 20, 30] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x + np.array([10, 20, 30], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 97,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-y-scalars",
            "name": "x + y {Scalars}",
            "parameters": {
                "function": lambda x, y: x + y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 27,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-y-tensor-2x3-and-scalar",
            "name": "x + y {Tensor of Shape 2x3 and Scalar}",
            "parameters": {
                "function": lambda x, y: x + y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 27,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-y-tensors-2x3",
            "name": "x + y {Tensors of Shape 2x3}",
            "parameters": {
                "function": lambda x, y: x + y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 27,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-plus-y-tensor-2x3-and-tensor-3",
            "name": "x + y {Tensor of Shape 2x3 and Vector of Size 3}",
            "parameters": {
                "function": lambda x, y: x + y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 27,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Subtraction
        {
            "id": "x-minus-24-scalar",
            "name": "x - 24 {Scalar}",
            "parameters": {
                "function": lambda x: x - 24,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 24,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "124-minus-x-scalar",
            "name": "124 - x {Scalar}",
            "parameters": {
                "function": lambda x: 124 - x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 124,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-24-tensor-2x3",
            "name": "x - 24 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x - 24,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 24,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "124-minus-x-tensor-2x3",
            "name": "124 - x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: 124 - x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 124,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-10-20-30-tensor-3",
            "name": "x - [10, 20, 30] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x - np.array([10, 20, 30], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 30,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "100-90-80-minus-x-tensor-3",
            "name": "[100, 90, 80] - x {Vector of Size 3}",
            "parameters": {
                "function": lambda x: np.array([100, 90, 80], dtype=np.uint8) - x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 80,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-10-20-30-tensor-2x3",
            "name": "x - [10, 20, 30] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x - np.array([10, 20, 30], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 30,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "100-90-80-minus-x-tensor-2x3",
            "name": "[100, 90, 80] - x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: np.array([100, 90, 80], dtype=np.uint8) - x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 80,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-y-scalars",
            "name": "x - y {Scalars}",
            "parameters": {
                "function": lambda x, y: x - y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 35,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 35,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-y-tensor-2x3-and-scalar",
            "name": "x - y {Tensor of Shape 2x3 and Scalar}",
            "parameters": {
                "function": lambda x, y: x - y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 35,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 35,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-y-tensors-2x3",
            "name": "x - y {Tensors of Shape 2x3}",
            "parameters": {
                "function": lambda x, y: x - y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 35,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 35,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-minus-y-tensor-2x3-and-tensor-3",
            "name": "x - y {Tensor of Shape 2x3 and Vector of Size 3}",
            "parameters": {
                "function": lambda x, y: x - y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 35,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 35,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Multiplication
        {
            "id": "x-times-7-scalar",
            "name": "x * 7 {Scalar}",
            "parameters": {
                "function": lambda x: x * 7,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 18,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-7-tensor-2x3",
            "name": "x * 7 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x * 7,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 18,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-1-2-3-tensor-3",
            "name": "x * [1, 2, 3] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x * np.array([1, 2, 3], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 42,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-1-2-3-tensor-2x3",
            "name": "x * [1, 2, 3] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x * np.array([1, 2, 3], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 42,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-y-scalars",
            "name": "x * y {Scalars}",
            "parameters": {
                "function": lambda x, y: x * y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 25,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-y-tensor-and-scalar",
            "name": "x * y {Tensor of Shape 2x3 and Scalar}",
            "parameters": {
                "function": lambda x, y: x * y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 25,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-y-tensor-and-scalar",
            "name": "x * y {Tensors of Shape 2x3}",
            "parameters": {
                "function": lambda x, y: x * y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 25,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-times-y-tensor-and-scalar",
            "name": "x * y {Tensor of Shape 2x3 and Vector of Size 3}",
            "parameters": {
                "function": lambda x, y: x * y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 25,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # True Division
        {
            "id": "x-truediv-10-scalar",
            "name": "x // 10 {Scalar}",
            "parameters": {
                "function": lambda x: x // 10,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "126-truediv-x-scalar",
            "name": "126 // x {Scalar}",
            "parameters": {
                "function": lambda x: 126 // x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 1,
                        "maximum": 126,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-10-tensor-2x3",
            "name": "x // 10 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x // 10,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "126-truediv-x-tensor-2x3",
            "name": "126 // x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: 126 // x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-5-10-15-tensor-3",
            "name": "x // [5, 10, 15] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x // np.array([5, 10, 15], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "120-60-30-truediv-x-tensor-3",
            "name": "[120, 60, 30] // x {Vector of Size 3}",
            "parameters": {
                "function": lambda x: np.array([120, 60, 30], dtype=np.uint8) // x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-5-10-15-tensor-2x3",
            "name": "x // [5, 10, 15] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x // np.array([5, 10, 15], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "120-60-30-truediv-x-tensor-2x3",
            "name": "[120, 60, 30] // x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: np.array([120, 60, 30], dtype=np.uint8) // x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-y-scalars",
            "name": "x // y {Scalars}",
            "parameters": {
                "function": lambda x, y: x // y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-y-tensor-2x3-and-scalar",
            "name": "x // y {Tensor of Shape 2x3 and Scalar}",
            "parameters": {
                "function": lambda x, y: x // y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-y-tensors-2x3",
            "name": "x // y {Tensors of Shape 2x3}",
            "parameters": {
                "function": lambda x, y: x // y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-truediv-y-tensor-2x3-and-tensor-3",
            "name": "x // y {Tensor of Shape 2x3 and Vector of Size 3}",
            "parameters": {
                "function": lambda x, y: x // y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 1,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Dot Product
        {
            "id": "x-dot-2-3-1-tensor-3",
            "name": "np.dot(x, [2, 3, 1]) {Vector of Size 3}",
            "parameters": {
                "function": lambda x: np.dot(x, np.array([2, 3, 1], dtype=np.uint8)),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "2-3-1-dot-x-tensor-3",
            "name": "np.dot([2, 3, 1], x) {Vector of Size 3}",
            "parameters": {
                "function": lambda x: np.dot(np.array([2, 3, 1], dtype=np.uint8), x),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-dot-y-tensors-3",
            "name": "np.dot(x, y) {Vectors of Size 3}",
            "parameters": {
                "function": lambda x, y: np.dot(x, y),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 14,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 3,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Matrix Multiplication
        {
            "id": "x-matmul-c-tensor-2x3",
            "name": "x @ [[1, 3], [3, 2], [2, 1]] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x @ np.array([[1, 3], [3, 2], [2, 1]], dtype=np.uint8),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "c-matmul-x-tensor-2x3",
            "name": "[[1, 3], [3, 2], [2, 1]] @ x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: np.array([[1, 3], [3, 2], [2, 1]], dtype=np.uint8) @ x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 30,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-matmul-y-tensor-2x3-and-tensor-3x2",
            "name": "x @ y {Tensor of Shape 2x3 and Tensor of Shape 3x2}",
            "parameters": {
                "function": lambda x, y: x @ y,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 14,
                    },
                    "y": {
                        "type": "encrypted",
                        "shape": (3, 2),
                        "minimum": 0,
                        "maximum": 3,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Negation
        {
            "id": "negative-x-plus-127-scalar",
            "name": "-x + 127 {Scalar}",
            "parameters": {
                "function": lambda x: -x + 127,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "negative-x-plus-127-tensor-2x3",
            "name": "-x + 127 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: -x + 127,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Power
        {
            "id": "x-to-the-power-of-2-scalar",
            "name": "x ** 2 {Scalar}",
            "parameters": {
                "function": lambda x: x ** 2,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 11,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "2-to-the-power-of-x-scalar",
            "name": "2 ** x {Scalar}",
            "parameters": {
                "function": lambda x: 2 ** x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 6,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-to-the-power-of-2-tensor-2x3",
            "name": "x ** 2 {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x ** 2,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 11,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "2-to-the-power-of-x-tensor-2x3",
            "name": "2 ** x {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: 2 ** x,
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 6,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Direct Table Lookup
        {
            "id": "single-table-lookup-5-bit-scalar",
            "name": "Single Table Lookup (5-Bit) {Scalar}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 5) % 32 for i in range(32)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 31,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "single-table-lookup-5-bit-tensor-2x3",
            "name": "Single Table Lookup (5-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 5) % 32 for i in range(32)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 31,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "multi-table-lookup-5-bit-tensor-2x3",
            "name": "Multi Table Lookup (5-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.MultiLookupTable(
                    [
                        [
                            hnp.LookupTable([(i ** 5) + 2 % 32 for i in range(32)]),
                            hnp.LookupTable([(i ** 5) * 3 % 32 for i in range(32)]),
                            hnp.LookupTable([(i ** 5) // 6 % 32 for i in range(32)]),
                        ],
                        [
                            hnp.LookupTable([(i ** 5) // 2 % 32 for i in range(32)]),
                            hnp.LookupTable([(i ** 5) + 5 % 32 for i in range(32)]),
                            hnp.LookupTable([(i ** 5) * 4 % 32 for i in range(32)]),
                        ],
                    ]
                )[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 31,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "single-table-lookup-6-bit-scalar",
            "name": "Single Table Lookup (6-Bit) {Scalar}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 6) % 64 for i in range(64)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 63,
                    },
                },
                "accuracy_alert_threshold": 99,
            },
        },
        {
            "id": "single-table-lookup-6-bit-tensor-2x3",
            "name": "Single Table Lookup (6-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 6) % 64 for i in range(64)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 63,
                    },
                },
                "accuracy_alert_threshold": 99,
            },
        },
        {
            "id": "multi-table-lookup-6-bit-tensor-2x3",
            "name": "Multi Table Lookup (6-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.MultiLookupTable(
                    [
                        [
                            hnp.LookupTable([(i ** 6) + 2 % 64 for i in range(64)]),
                            hnp.LookupTable([(i ** 6) * 3 % 64 for i in range(64)]),
                            hnp.LookupTable([(i ** 6) // 6 % 64 for i in range(64)]),
                        ],
                        [
                            hnp.LookupTable([(i ** 6) // 2 % 64 for i in range(64)]),
                            hnp.LookupTable([(i ** 6) + 5 % 64 for i in range(64)]),
                            hnp.LookupTable([(i ** 6) * 4 % 64 for i in range(64)]),
                        ],
                    ]
                )[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 63,
                    },
                },
                "accuracy_alert_threshold": 99,
            },
        },
        {
            "id": "single-table-lookup-7-bit-scalar",
            "name": "Single Table Lookup (7-Bit) {Scalar}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 7) % 128 for i in range(128)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 95,
            },
        },
        {
            "id": "single-table-lookup-7-bit-tensor-2x3",
            "name": "Single Table Lookup (7-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.LookupTable([(i ** 7) % 128 for i in range(128)])[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 95,
            },
        },
        {
            "id": "multi-table-lookup-7-bit-tensor-2x3",
            "name": "Multi Table Lookup (7-Bit) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: hnp.MultiLookupTable(
                    [
                        [
                            hnp.LookupTable([(i ** 7) + 2 % 128 for i in range(128)]),
                            hnp.LookupTable([(i ** 7) * 3 % 128 for i in range(128)]),
                            hnp.LookupTable([(i ** 7) // 6 % 128 for i in range(128)]),
                        ],
                        [
                            hnp.LookupTable([(i ** 7) // 2 % 128 for i in range(128)]),
                            hnp.LookupTable([(i ** 7) + 5 % 128 for i in range(128)]),
                            hnp.LookupTable([(i ** 7) * 4 % 128 for i in range(128)]),
                        ],
                    ]
                )[x],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 95,
            },
        },
        # Manipulation
        {
            "id": "transpose-tensor-2x3",
            "name": "np.transpose(x) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: np.transpose(x),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "reshape-to-1-3-1-2-1-tensor-2x3",
            "name": "np.reshape(x, (1, 3, 1, 2, 1)) {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: np.reshape(x, (1, 3, 1, 2, 1)),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (1, 3, 1, 2, 1),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "flatten-tensor-2x3",
            "name": "x.flatten() {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x.flatten(),
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Indexing
        {
            "id": "x-index-0-tensor-3",
            "name": "x[0] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[0],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-1-tensor-3",
            "name": "x[1] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-2-tensor-3",
            "name": "x[2] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[2],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-minus-1-tensor-3",
            "name": "x[-1] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[-1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-minus-2-tensor-3",
            "name": "x[-2] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[-2],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-minus-3-tensor-3",
            "name": "x[-3] {Vector of Size 3}",
            "parameters": {
                "function": lambda x: x[-3],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-0-and-0-tensor-2x3",
            "name": "x[0, 0] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x[0, 0],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-minus-1-and-minus-1-tensor-2x3",
            "name": "x[-1, -1] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x[-1, -1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-0-tensor-2x3",
            "name": "x[0] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x[0],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-minus-1-tensor-2x3",
            "name": "x[-1] {Tensor of Shape 2x3}",
            "parameters": {
                "function": lambda x: x[-1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (2, 3),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-y-tensor-5-and-scalar",
            "name": "x[y] {Vector of Size 5 and Scalar}",
            "parameters": {
                "function": lambda x, y: x[y],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 4,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-y-and-z-tensor-5-and-scalars",
            "name": "x[y] {Tensor of Shape 5x3 and Scalars}",
            "parameters": {
                "function": lambda x, y, z: x[y, z],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                    "y": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 4,
                    },
                    "z": {
                        "type": "encrypted",
                        "minimum": 0,
                        "maximum": 2,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        # Slicing
        {
            "id": "x-reversed-tensor-5",
            "name": "x[::-1] {Vector of Size 5}",
            "parameters": {
                "function": lambda x: x[::-1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-colon-tensor-5",
            "name": "x[:] {Vector of Size 5}",
            "parameters": {
                "function": lambda x: x[:],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-2-colon-tensor-5",
            "name": "x[2:] {Vector of Size 5}",
            "parameters": {
                "function": lambda x: x[2:],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-colon-3-tensor-5",
            "name": "x[:3] {Vector of Size 5}",
            "parameters": {
                "function": lambda x: x[:3],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-1-colon-3-tensor-5",
            "name": "x[1:3] {Vector of Size 5}",
            "parameters": {
                "function": lambda x: x[1:3],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (5,),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-colon-and-1-tensor-3x2",
            "name": "x[:, 1] {Tensor of Shape 3x2}",
            "parameters": {
                "function": lambda x: x[:, 1],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (3, 2),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
        {
            "id": "x-index-1-colon-3-and-1-colon-3-tensor-4x4",
            "name": "x[1:3, 1:3] {Tensor of Shape 4x4}",
            "parameters": {
                "function": lambda x: x[1:3, 1:3],
                "inputs": {
                    "x": {
                        "type": "encrypted",
                        "shape": (4, 4),
                        "minimum": 0,
                        "maximum": 127,
                    },
                },
                "accuracy_alert_threshold": 100,
            },
        },
    ]
)
def main(function, inputs, accuracy_alert_threshold):
    inputset = []
    for _ in range(128):
        input_ = []

        for description in inputs.values():
            minimum = description["minimum"]
            maximum = description["maximum"]

            assert minimum >= 0
            assert maximum <= 127

            if "shape" in description:
                shape = description["shape"]
                input_.append(np.random.randint(minimum, maximum + 1, size=shape, dtype=np.uint8))
            else:
                input_.append(random.randint(minimum, maximum))

        inputset.append(tuple(input_) if len(input_) > 1 else input_[0])

    compiler = hnp.NPFHECompiler(
        function, {name: description["type"] for name, description in inputs.items()}
    )

    circuit = compiler.compile_on_inputset(inputset)

    samples = []
    expectations = []
    for _ in range(128):
        sample = []
        for description in inputs.values():
            minimum = description["minimum"]
            maximum = description["maximum"]

            assert minimum >= 0
            assert maximum <= 127

            if "shape" in description:
                shape = description["shape"]
                sample.append(np.random.randint(minimum, maximum + 1, size=shape, dtype=np.uint8))
            else:
                sample.append(random.randint(minimum, maximum))

        samples.append(sample)
        expectations.append(function(*sample))

    correct = 0
    for sample_i, expectation_i in zip(samples, expectations):
        with progress.measure(id="evaluation-time-ms", label="Evaluation Time (ms)"):
            result_i = circuit.run(*sample_i)

        if np.array_equal(result_i, expectation_i):
            correct += 1
    accuracy = (correct / len(samples)) * 100

    print(f"Accuracy (%): {accuracy:.4f}")
    progress.measure(
        id="accuracy-percent",
        label="Accuracy (%)",
        value=accuracy,
        alert=("<", accuracy_alert_threshold),
    )
