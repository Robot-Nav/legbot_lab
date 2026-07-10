# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UI 扩展示例，展示如何在 Omniverse 扩展中创建简单窗口。"""

import omni.ext


def some_public_function(x: int):
    """示例公共函数，计算 x 的 x 次幂。

    参数:
        x: 输入整数。

    返回:
        x 的 x 次幂。
    """
    print('[robot_lab] some_public_function 被调用，x: ', x)
    return x**x


class ExampleExtension(omni.ext.IExt):
    """Omniverse 扩展示例类，在扩展启用时创建计数器窗口。"""

    def on_startup(self, ext_id):
        """扩展启动时调用，创建 UI 窗口并绑定按钮事件。

        参数:
            ext_id: 当前扩展 ID，可用于查询扩展信息。
        """
        print('[robot_lab] 启动')

        self._count = 0

        self._window = omni.ui.Window('计数窗口', width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                label = omni.ui.Label('')

                def on_click():
                    self._count += 1
                    label.text = f'计数: {self._count}'

                def on_reset():
                    self._count = 0
                    label.text = '空'

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button('增加', clicked_fn=on_click)
                    omni.ui.Button('重置', clicked_fn=on_reset)

    def on_shutdown(self):
        """扩展关闭时调用。"""
        print('[robot_lab] 关闭')
