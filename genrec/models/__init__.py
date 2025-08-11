# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = []

# 使用try-except来处理可能缺失的模块
try:
    from genrec.models.DIFF_GRM.model import DIFF_GRM
except ImportError:
    DIFF_GRM = None
    print("Warning: DIFF_GRM model not found, skipping import")

try:
    from genrec.models.AR_GRM.model import AR_GRM
except ImportError:
    AR_GRM = None
    print("Warning: AR_GRM model not found, skipping import")


# 只导出存在的模型
if DIFF_GRM is not None:
    __all__.append('DIFF_GRM')
if AR_GRM is not None:
    __all__.append('AR_GRM')

