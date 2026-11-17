"""构建独立 V2.1 修订训练版本，保留 V1/V2 文件及固定 V2 评测面板。"""
import json
from _bootstrap import PROJECT_ROOT
from src.data.diagnostic import prepare_diagnostic_release

if __name__ == '__main__':
    # 旧版只保留冻结评测的可复现性；训练配置明确指向修正的 V2.1。
    for revision in (1,2):
        result=prepare_diagnostic_release(PROJECT_ROOT,notation_revision=revision)
        print(json.dumps({'notation_revision':revision,'summary':result},ensure_ascii=False,indent=2))
