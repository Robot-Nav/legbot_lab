# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 包入口：导出当前激活的 Section001 导航环境类及其配置，便于外部按名称注册调用。

# from . import legbot_np, legbot_stairs_np, legbot_stairs_multi_target_np, legbot_long_course_np, cfg # noqa: F401
from . import legbot_section001_np, cfg  # noqa: F401
from .legbot_section001_np import VBotSection001Env  # Section001 导航环境实现
# 以下环境当前未启用，保留供后续扩展：
# from .legbot_np import VBotEnv  # noqa: F401
# from .legbot_stairs_np import VBotStairsEnv  # noqa: F401
# from .legbot_stairs_multi_target_np import VBotStairsMultiTargetEnv  # noqa: F401
# from .legbot_long_course_np import VBotLongCourseEnv  # noqa: F401
# from .legbot_section01_np import VBotSection01Env
# from .legbot_section02_np import VBotSection02Env
# from .legbot_section03_np import VBotSection03Env
from .cfg import (  # noqa: F401
    VBotEnvCfg,
    VBotStairsEnvCfg,
    VBotSection01EnvCfg,
    VBotSection02EnvCfg,
    VBotSection03EnvCfg,
    VBotLongCourseEnvCfg,
    VBotSection001EnvCfg,
)
