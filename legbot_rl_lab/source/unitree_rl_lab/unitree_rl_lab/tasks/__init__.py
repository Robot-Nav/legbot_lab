##
# 注册 Gym 环境，并自动导入本包及其子包中的配置。
##

from isaaclab_tasks.utils import import_packages

# 黑名单用于禁止从特定子包导入配置
_BLACKLIST_PKGS = []
# 导入本包中的所有配置
import_packages(__name__, _BLACKLIST_PKGS)
