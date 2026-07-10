# Omniverse 扩展 UI 示例：展示如何在扩展启用时创建简单计数器窗口。
import omni.ext


# 其他扩展可通过常规 Python 方式访问该函数：example.python_ext.some_public_function(x)
def some_public_function(x: int):
    print("[unitree_rl_lab] some_public_function 被调用，参数 x: ", x)
    return x**x


# 任何派生自 omni.ext.IExt 的顶层类，在扩展启用时会实例化并调用 on_startup(ext_id)。
# 扩展禁用时调用 on_shutdown()。
class ExampleExtension(omni.ext.IExt):
    # ext_id 为当前扩展 ID，可用于查询扩展在文件系统中的位置等信息。
    def on_startup(self, ext_id):
        print("[unitree_rl_lab] startup")

        self._count = 0

        self._window = omni.ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            with omni.ui.VStack():
                label = omni.ui.Label("")

                def on_click():
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    self._count = 0
                    label.text = "empty"

                on_reset()

                with omni.ui.HStack():
                    omni.ui.Button("Add", clicked_fn=on_click)
                    omni.ui.Button("Reset", clicked_fn=on_reset)

    def on_shutdown(self):
        print("[unitree_rl_lab] shutdown")
