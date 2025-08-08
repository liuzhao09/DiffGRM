# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 使用try-except来处理可能缺失的模块
try:
    from genrec.models.RPG.model import RPG
except ImportError:
    RPG = None
    print("Warning: RPG model not found, skipping import")

try:
    from genrec.models.RPG_ED.model import RPG_ED
except ImportError:
    RPG_ED = None
    print("Warning: RPG_ED model not found, skipping import")

try:
    from genrec.models.DIFF_GRM.model import DIFF_GRM
except ImportError:
    DIFF_GRM = None
    print("Warning: DIFF_GRM model not found, skipping import")

try:
    from genrec.models.DIFF_GRM_V1.model import DIFF_GRM_V1
except ImportError:
    DIFF_GRM_V1 = None
    print("Warning: DIFF_GRM_V1 model not found, skipping import")

# 只导出存在的模型
__all__ = []
if RPG is not None:
    __all__.append('RPG')
if RPG_ED is not None:
    __all__.append('RPG_ED')
if DIFF_GRM is not None:
    __all__.append('DIFF_GRM')
if DIFF_GRM_V1 is not None:
    __all__.append('DIFF_GRM_V1')
