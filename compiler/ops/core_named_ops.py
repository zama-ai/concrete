from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2

Batch = S.Batch


@linalg_structured_op
def fhelinalg_conv_2d_nchw_fchw(
    I=TensorDef(T1, S.N, S.C, S.OH * S.SH + S.KH * S.DH, S.OW * S.SW + S.KW * S.DW),
    K=TensorDef(T2, S.F, S.C, S.KH, S.KW),
    O=TensorDef(U, S.N, S.F, S.OH, S.OW, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  """Performs 2-D convolution.

  Layout:
    * Input: NCHW.
    * Kernel: FCHW.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  implements(ConvolutionOpInterface)
  domain(D.n, D.f, D.oh, D.ow, D.c, D.kh, D.kw)
  O[D.n, D.f, D.oh, D.ow] += cast(
      U, I[D.n, D.c, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW
           ]) * cast(U, K[D.f, D.c, D.kh, D.kw])