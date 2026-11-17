"""全局最优多目标关联求解器（野生动物追踪串号校正）。

问题模型
--------
- 每个检测框恰好属于一条轨迹；
- 同一轨迹在同一帧至多一个框；
- 轨迹内相邻两框的帧差为 1 或 2（时间严格递增）；
- 关联代价 = w_dist * 框中心距离平方 + w_app * (1 - 余弦相似度)；
- 目标 = 所有关联代价之和 + 轨迹条数 * 每条新轨迹罚分，求全局最小。

求解方法
--------
整数线性规划（SciPy `milp` / HiGHS，可证明的全局最优）：

- ``x_e ∈ {0,1}``：是否选用有向边 e（连接帧差 1~2 的两个框）；
- ``s_i ∈ {0,1}``：框 i 是否为轨迹起点，轨迹条数 = Σ s_i；
- 度约束：``s_i + 入度(i) = 1``，``出度(i) ≤ 1``
  （时间递增的有向图无环，故选中边构成点不相交路径覆盖）；
- “同只”判断：对同只等价类 C（并查集），取其最早框 r，
  对每个其他成员 m 引入单位流变量 ``0 ≤ y_e ≤ x_e``，
  要求从 r 到 m 输送 1 单位流 —— 即 m 在选中子图中可由 r 到达。
  由于每点出/入度 ≤ 1，可达 ⇔ 在同一条轨迹上；
- “异只”判断：对每对 (a, b)（帧 a < 帧 b），引入 0/1 割变量 π，
  ``π_a = 1, π_b = 0, π_u + x_uv - π_v ≤ 1``，
  即可达集 S（π=1）没有选中边流出 ⇔ b 不可达 a ⇔ 不同轨迹。

若 ILP 不可行（判断相互冲突），明确报告无解，不给出计数。

冲突定位
--------
``locate_conflict`` 在判断条数不超过 8 条、且已确认无解时，用删除法求一个
**最小不可行判断组**（MIS）：该组单独即不可行，删去组内任意一条则可行。
每一次可行性检查都带时限；任何一次检查超时或出错都抛出 ``LocationIncomplete``，
绝不猜测冲突。可行性只取决于身份判断，与代价参数无关。
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# milp 状态码（HiGHS）
_STATUS_OPTIMAL = 0
_STATUS_INFEASIBLE = 2

# 冲突定位：条数上限、单次可行性检查时限、整次定位总时限（秒）
MAX_LOCATE_JUDGMENTS = 8
FEASIBILITY_TIME_LIMIT = 5.0
LOCATE_TOTAL_TIME_LIMIT = 30.0


class InputConflict(Exception):
    """输入层面即可判定的判断冲突（无需求解即可报告具体原因）。"""


class LocationIncomplete(Exception):
    """冲突定位时某次可行性检查超时或出错；应提示定位未完成，不猜测冲突。"""


def _norm_box(d):
    """返回 (id, frame, center(x,y), vector) 的规范化视图。"""
    x, y, w, h = (float(v) for v in d["rect"])
    vec = np.asarray(d["vector"], dtype=float)
    return str(d["id"]), int(d["frame"]), (x + w / 2.0, y + h / 2.0), vec


def _build_edges(detections, w_dist, w_app):
    """构造所有帧差 1~2 的候选边及其代价。返回 [(i, j, cost), ...]"""
    n = len(detections)
    info = [_norm_box(d) for d in detections]
    frames = [t[1] for t in info]
    centers = np.array([t[2] for t in info], dtype=float)
    vecs = [t[3] for t in info]
    norms = [float(np.linalg.norm(v)) for v in vecs]

    edges = []
    for i in range(n):
        for j in range(n):
            gap = frames[j] - frames[i]
            if gap < 1 or gap > 2:
                continue
            d2 = float(np.sum((centers[i] - centers[j]) ** 2))
            cos = float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j]))
            cos = max(-1.0, min(1.0, cos))  # 数值截断
            cost = w_dist * d2 + w_app * (1.0 - cos)
            edges.append((i, j, cost))
    return edges, frames


def _same_classes(n, judgments, frames, ids):
    """并查集合并同只判断；返回 (classes, diff_pairs)。

    classes: list[list[int]] 仅含成员数 > 1 的等价类（按帧升序）。
    diff_pairs: list[(a, b)] 帧 a < 帧 b 的异只约束（同帧的自动满足，丢弃）。
    发现显然冲突时抛出 InputConflict。
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    raw_diff = []
    for j in judgments:
        a, b = j["a"], j["b"]
        if j["type"] == "same":
            union(a, b)
        else:
            raw_diff.append((a, b))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    classes = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda i: frames[i])
        seen = {}
        for m in members:
            f = frames[m]
            if f in seen:
                raise InputConflict(
                    f"框 {ids[seen[f]]} 与框 {ids[m]} 被（传递地）判定为同只，"
                    f"但二者都在第 {f} 帧；同一轨迹每帧至多一个框，约束冲突。"
                )
            seen[f] = m
        classes.append(members)

    diff_pairs = []
    for a, b in raw_diff:
        if find(a) == find(b):
            raise InputConflict(
                f"框 {ids[a]} 与框 {ids[b]} 既被判定为同只（可能经传递）又被判定为异只，约束冲突。"
            )
        if frames[a] == frames[b]:
            continue  # 同帧必然不同轨迹，约束自动满足
        if frames[a] > frames[b]:
            a, b = b, a
        diff_pairs.append((a, b))
    return classes, diff_pairs


def _build_model(detections, judgments, w_dist, w_app, penalty):
    """构造 ILP 模型（与求解/可行性检查共用）。

    返回字典，包含变量布局、约束矩阵、变量界限以及按关联代价构造的目标向量。
    输入层面的显然冲突抛出 InputConflict。
    """
    n = len(detections)
    ids = [str(d["id"]) for d in detections]
    if n == 0:
        return {"n": 0, "ids": ids, "frames": [], "edges": [], "model": None}

    edges, frames = _build_edges(detections, w_dist, w_app)
    classes, diff_pairs = _same_classes(n, judgments, frames, ids)

    E = len(edges)
    # 同只流商品：(root, member) 对每个非平凡类的每个非根成员
    commodities = []
    for members in classes:
        root = members[0]
        for m in members[1:]:
            commodities.append((root, m))
    F = len(commodities)
    D = len(diff_pairs)

    # 变量布局: x[E] | s[n] | y[F*E] | pi[D*n]
    off_x = 0
    off_s = off_x + E
    off_y = off_s + n
    off_pi = off_y + F * E
    n_vars = off_pi + D * n

    c = np.zeros(n_vars)
    for k, (_, _, cost) in enumerate(edges):
        c[off_x + k] = cost
    c[off_s : off_s + n] = penalty

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[off_y : off_y + F * E] = 1.0  # y ≤ x ≤ 1
    integrality = np.zeros(n_vars, dtype=int)
    integrality[off_x : off_s + n] = 1  # x, s 为 0/1
    integrality[off_pi:] = 1  # pi 为 0/1

    rows, cols, vals = [], [], []
    lb_c, ub_c = [], []
    n_rows = 0

    def add_row(entries, lo, hi):
        nonlocal n_rows
        for col, val in entries:
            rows.append(n_rows)
            cols.append(col)
            vals.append(val)
        lb_c.append(lo)
        ub_c.append(hi)
        n_rows += 1

    in_edges = [[] for _ in range(n)]
    out_edges = [[] for _ in range(n)]
    for k, (i, j, _) in enumerate(edges):
        out_edges[i].append(k)
        in_edges[j].append(k)

    inf = math.inf
    # 1) s_i + 入度(i) = 1
    for i in range(n):
        add_row([(off_s + i, 1.0)] + [(off_x + k, 1.0) for k in in_edges[i]], 1.0, 1.0)
    # 2) 出度(i) ≤ 1
    for i in range(n):
        if out_edges[i]:
            add_row([(off_x + k, 1.0) for k in out_edges[i]], -inf, 1.0)
    # 3) 同只可达流：每商品每节点流量守恒
    for q, (root, member) in enumerate(commodities):
        base = off_y + q * E
        for v in range(n):
            target = 0.0
            if v == root:
                target = 1.0
            elif v == member:
                target = -1.0
            entries = [(base + k, 1.0) for k in out_edges[v]]
            entries += [(base + k, -1.0) for k in in_edges[v]]
            if entries:
                add_row(entries, target, target)
            elif target != 0.0:
                # 该商品源/汇无任何边可用 → 必然不可行
                raise InputConflict(
                    f"框 {ids[root]} 与框 {ids[member]} 被判定为同只，"
                    "但二者之间不存在帧差 1~2 的连接路径，约束冲突。"
                )
        # 4) y ≤ x
        for k in range(E):
            add_row([(base + k, 1.0), (off_x + k, -1.0)], -inf, 0.0)
    # 5) 异只割约束
    for d, (a, b) in enumerate(diff_pairs):
        base = off_pi + d * n
        add_row([(base + a, 1.0)], 1.0, 1.0)  # pi_a = 1
        add_row([(base + b, 1.0)], 0.0, 0.0)  # pi_b = 0
        for k, (u, v, _) in enumerate(edges):
            add_row([(base + u, 1.0), (base + v, -1.0), (off_x + k, 1.0)], -inf, 1.0)

    A = coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_rows, n_vars),
    ).tocsr()
    constraints = LinearConstraint(A, np.array(lb_c), np.array(ub_c))

    return {
        "n": n,
        "ids": ids,
        "frames": frames,
        "edges": edges,
        "model": {
            "c": c,
            "constraints": constraints,
            "bounds": Bounds(lb, ub),
            "integrality": integrality,
            "off_x": off_x,
            "off_s": off_s,
        },
    }


def _run_milp(model, objective, time_limit):
    return milp(
        c=objective,
        constraints=model["constraints"],
        bounds=model["bounds"],
        integrality=model["integrality"],
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )


def _solve_impl(detections, judgments, w_dist, w_app, penalty, time_limit=60.0):
    n = len(detections)
    ids = [str(d["id"]) for d in detections]
    if n == 0:
        return {
            "feasible": True,
            "message": "无检测框。",
            "total_cost": 0.0,
            "assoc_cost": 0.0,
            "penalty_cost": 0.0,
            "n_trajectories": 0,
            "animal_count": 0,
            "trajectories": [],
        }

    built = _build_model(detections, judgments, w_dist, w_app, penalty)
    edges = built["edges"]
    frames = built["frames"]
    mdl = built["model"]
    res = _run_milp(mdl, mdl["c"], time_limit)

    if res.status == _STATUS_INFEASIBLE:
        return {
            "feasible": False,
            "reason": "infeasible",
            "message": "约束不可满足：身份判断相互冲突（全局最优解不存在），不提供动物计数。",
        }
    if res.status != _STATUS_OPTIMAL or res.x is None:
        return {
            "feasible": False,
            "reason": "solver_error",
            "message": f"求解未收敛到全局最优（求解器状态：{res.message}），不提供动物计数。",
        }

    x = res.x
    off_x, off_s = mdl["off_x"], mdl["off_s"]
    succ = {}
    assoc_cost = 0.0
    for k, (i, j, cost) in enumerate(edges):
        if x[off_x + k] > 0.5:
            succ[i] = j
            assoc_cost += cost
    starts = [i for i in range(n) if x[off_s + i] > 0.5]

    trajectories = []
    for s in starts:
        traj = [s]
        while traj[-1] in succ:
            traj.append(succ[traj[-1]])
        trajectories.append(traj)
    trajectories.sort(key=lambda t: (frames[t[0]], ids[t[0]]))

    n_traj = len(trajectories)
    penalty_cost = penalty * n_traj
    return {
        "feasible": True,
        "message": "求解成功（全局最优）。",
        "total_cost": float(res.fun),
        "assoc_cost": float(assoc_cost),
        "penalty_cost": float(penalty_cost),
        "n_trajectories": n_traj,
        "animal_count": n_traj,
        "trajectories": [
            {
                "boxes": [
                    {"id": ids[i], "frame": frames[i]} for i in traj
                ],
                "length": len(traj),
            }
            for traj in trajectories
        ],
    }


def solve_tracking(detections, judgments, w_dist, w_app, penalty, time_limit=60.0):
    """求全局最小关联。返回结果字典（含 feasible 标志）。

    detections: [{"id":..,"frame":int,"rect":[x,y,w,h],"vector":[...]}]
    judgments:  [{"a":idx,"b":idx,"type":"same"|"different"}]（a、b 为 detections 下标）
    """
    try:
        return _solve_impl(detections, judgments, w_dist, w_app, penalty, time_limit)
    except InputConflict as exc:
        return {
            "feasible": False,
            "reason": "infeasible",
            "message": f"无解：{exc} 不提供动物计数。",
        }


def _check_feasible(detections, judgments, w_dist, w_app, penalty, time_limit):
    """判断给定判断组合下是否**存在**可行关联。

    返回 True/False。输入层面的显然冲突（InputConflict）是确定性不可行证明，
    返回 False；求解器未在时限内给出结论或抛出其它异常时抛 LocationIncomplete，
    绝不把“不知道”当成不可行。
    """
    try:
        built = _build_model(detections, judgments, w_dist, w_app, penalty)
    except InputConflict:
        return False
    if built["n"] == 0:
        return True
    mdl = built["model"]
    zero = np.zeros_like(mdl["c"])
    try:
        res = _run_milp(mdl, zero, time_limit)
    except Exception as exc:  # noqa: BLE001 - 定位必须显式报告未完成
        raise LocationIncomplete(f"可行性检查出错：{exc}") from exc
    if res.status == _STATUS_INFEASIBLE:
        return False
    if res.status == _STATUS_OPTIMAL and res.x is not None:
        return True
    raise LocationIncomplete(
        f"可行性检查未完成（求解器状态：{res.message}）。"
    )


def locate_conflict(detections, judgments, w_dist, w_app, penalty,
                    check_time_limit=FEASIBILITY_TIME_LIMIT,
                    total_time_limit=LOCATE_TOTAL_TIME_LIMIT):
    """定位一组导致无解的最小身份判断。

    judgments 的 a/b 为 detections 下标；返回的 ``judgments`` 保持传入顺序中的
    具体条目（子集）。成功后保证：
    - 该子集单独即不可行；
    - 删除子集中任意一条后即可行。
    任何一次可行性检查超时/出错 → 返回 status="incomplete"，不猜测冲突。
    """
    if len(judgments) > MAX_LOCATE_JUDGMENTS:
        return {
            "status": "too_many",
            "message": (
                f"当前有 {len(judgments)} 条判断，超过 {MAX_LOCATE_JUDGMENTS} 条，"
                "暂不提供冲突定位；仍可正常求解。"
            ),
            "limit": MAX_LOCATE_JUDGMENTS,
        }

    deadline = time.monotonic() + total_time_limit
    checks = 0

    def feasible(subset):
        nonlocal checks
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LocationIncomplete("定位总时长已用尽。")
        checks += 1
        try:
            return _check_feasible(
                detections, subset, w_dist, w_app, penalty,
                min(check_time_limit, remaining),
            )
        except LocationIncomplete as exc:
            raise LocationIncomplete(f"第 {checks} 次可行性检查超时或出错：{exc}") from exc

    try:
        if feasible(judgments):
            return {
                "status": "feasible",
                "checks": checks,
                "message": "当前判断组合本身可行，未发现可定位的判断冲突。",
            }

        # 删除法：逐条尝试去掉；去后仍不可行则永久去掉。
        current = list(range(len(judgments)))
        for idx in range(len(judgments)):
            if idx not in current:
                continue
            trial = [i for i in current if i != idx]
            if not feasible([judgments[i] for i in trial]):
                current = trial

        group = [judgments[i] for i in current]

        # 显式复核交付给用户的保证：组本身不可行，且去掉任意一条即可行。
        if feasible(group):
            raise LocationIncomplete("复核冲突组时结果不一致（组本身反而可行），未给出冲突判断。")
        for idx in current:
            trial = [judgments[i] for i in current if i != idx]
            if not feasible(trial):
                raise LocationIncomplete(
                    "复核冲突组时结果不一致（并非最小冲突组），未给出冲突判断。"
                )
    except LocationIncomplete as exc:
        return {
            "status": "incomplete",
            "checks": checks,
            "message": f"冲突定位未完成：{exc} 不猜测冲突判断。",
        }
    except Exception as exc:  # noqa: BLE001 - 任何异常都不能变成猜测
        return {
            "status": "incomplete",
            "checks": checks,
            "message": f"冲突定位未完成：定位过程出错（{exc}）。不猜测冲突判断。",
        }

    return {
        "status": "found",
        "checks": checks,
        "judgments": group,
        "message": (
            f"已定位 {len(group)} 条冲突判断：单独这组即使轨迹不可行；"
            "删除其中任意一条，轨迹即可行。"
        ),
    }
