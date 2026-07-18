"""
Domain 3: 認知・生命系（意味ネットワークの再構成 / 能動推論による世界解釈）

オブジェクトは「知覚・概念・シンボル」
関係 R は「意味の隣接関係（コンテキスト）」

外部からの刺激（差異）によって、概念間のリンク（関係）がドラスティックに書き換わる。
不変量（自由エネルギー最小化）すら必要なく、ただ「関係の遷移規則」として認知の発展を規定する。
"""

from typing import Any, Set, Tuple

Object = Any
Relation = Set[Tuple[Object, Object]]


def cognitive_evolution(R: Relation) -> Relation:
    """Cognitive evolution through meaning network reconstruction."""
    # 外部からの刺激（差異）によって、概念間のリンク（関係）がドラスティックに書き換わる
    return {(src, dst) for (src, dst) in R if src != "old_dogma"} | {("new_sensory", "understanding")}
