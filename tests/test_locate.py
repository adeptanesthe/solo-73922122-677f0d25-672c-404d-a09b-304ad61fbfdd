"""冲突定位测试：最小冲突组的正确性（单独不可行、删任一条即可行）、
条数上限、可行性结论，以及超时/出错时绝不猜测的失败路径。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.solver as solver
from app.solver import solve_tracking
from tests.test_solver import PENALTY, W_APP, W_DIST, mk


def is_infeasible(dets, judgments):
    return solve_tracking(dets, judgments, W_DIST, W_APP, PENALTY)["feasible"] is False


def check_minimal(dets, all_judgments, group):
    """验证定位保证：组单独不可行，删去组内任意一条则可行。"""
    assert group, "冲突组不应为空"
    assert is_infeasible(dets, group), "冲突组单独应不可行"
    for idx in range(len(group)):
        assert not is_infeasible(dets, group[:idx] + group[idx + 1:]), (
            f"删去第 {idx} 条后应可行"
        )


def chain_dets():
    return [
        mk("a", 0, 10, 10, [1, 0]),
        mk("b", 1, 20, 10, [1, 0]),
        mk("c", 2, 30, 10, [1, 0]),
        mk("d", 3, 40, 10, [1, 0]),
    ]


def test_locate_transitive_conflict_is_minimal():
    dets = chain_dets()
    judgments = [
        {"a": 0, "b": 1, "type": "same"},
        {"a": 1, "b": 2, "type": "same"},
        {"a": 0, "b": 2, "type": "different"},
        {"a": 2, "b": 3, "type": "same"},
    ]
    assert is_infeasible(dets, judgments)
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "found"
    assert len(r["judgments"]) == 3  # 与无解无关的 b2-b3 同只不得混入
    check_minimal(dets, judgments, r["judgments"])
    assert all(j in judgments for j in r["judgments"])


def test_locate_single_same_frame_conflict():
    dets = [mk("a", 0, 0, 0, [1, 0]), mk("b", 0, 50, 0, [1, 0]),
            mk("c", 1, 60, 0, [1, 0])]
    judgments = [
        {"a": 1, "b": 2, "type": "same"},
        {"a": 0, "b": 1, "type": "same"},
    ]
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "found"
    assert r["judgments"] == [{"a": 0, "b": 1, "type": "same"}]
    check_minimal(dets, judgments, r["judgments"])


def test_locate_gap3_same_conflict():
    dets = [mk("a", 0, 10, 10, [1, 0]), mk("b", 3, 40, 10, [1, 0])]
    judgments = [{"a": 0, "b": 1, "type": "same"}]
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "found" and len(r["judgments"]) == 1
    check_minimal(dets, judgments, r["judgments"])


def test_locate_reports_feasible_when_solvable():
    dets = chain_dets()
    judgments = [{"a": 0, "b": 1, "type": "same"}]
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "feasible"


def test_locate_too_many_judgments():
    dets = chain_dets()
    judgments = [{"a": 0, "b": 1, "type": "same"}] * (solver.MAX_LOCATE_JUDGMENTS + 1)
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "too_many" and r["limit"] == solver.MAX_LOCATE_JUDGMENTS


def test_locate_at_limit_still_works():
    """恰好 8 条判断仍应定位：包含一对同帧框同只的传递冲突。"""
    # b0/b1/b2 都是第 0 帧；仅当把 b0-b1 链与 b2 串进同一等价类时冲突
    dets = [
        mk("b0", 0, 0, 0, [1, 0]),
        mk("b1", 1, 10, 0, [1, 0]),
        mk("b2", 0, 20, 0, [1, 0]),
        mk("b3", 1, 30, 0, [1, 0]),
    ]
    judgments = [
        {"a": 0, "b": 1, "type": "same"},   # 无冲突的同只
        {"a": 2, "b": 3, "type": "same"},   # 无冲突的同只
        {"a": 0, "b": 2, "type": "different"},
        {"a": 0, "b": 3, "type": "different"},
        {"a": 1, "b": 2, "type": "different"},
        {"a": 1, "b": 3, "type": "different"},
        {"a": 0, "b": 2, "type": "same"},   # 同帧同只 → 关键冲突
        {"a": 1, "b": 3, "type": "same"},
    ]
    # 直接传入的 judgments 可能互相重复（API 会合并），这里保持 8 条
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "found", r
    check_minimal(dets, judgments, r["judgments"])


def test_locate_checks_params_do_not_matter():
    """可行性只取决于身份判断，与代价参数无关。"""
    dets = chain_dets()
    judgments = [
        {"a": 0, "b": 1, "type": "same"},
        {"a": 1, "b": 2, "type": "same"},
        {"a": 0, "b": 2, "type": "different"},
    ]
    a = solver.locate_conflict(dets, judgments, 0.0, 0.0, 0.0)
    b = solver.locate_conflict(dets, judgments, 5.0, 3.0, 1e9)
    assert a["status"] == b["status"] == "found"
    assert a["judgments"] == b["judgments"]


def test_locate_solver_exception_is_incomplete_not_guess(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated timeout")

    monkeypatch.setattr(solver, "_run_milp", boom)
    dets = chain_dets()
    judgments = [{"a": 0, "b": 1, "type": "same"}]
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "incomplete"
    assert "judgments" not in r
    assert "未完成" in r["message"]


def test_locate_solver_nonfinal_status_is_incomplete(monkeypatch):
    class NotFinal:
        status = 1  # 非最优、非不可行（如时限）
        message = "time limit reached"
        x = None

    monkeypatch.setattr(solver, "_run_milp", lambda *a, **k: NotFinal())
    dets = chain_dets()
    judgments = [{"a": 0, "b": 1, "type": "same"}]
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    assert r["status"] == "incomplete"
    assert "judgments" not in r


def test_locate_budget_exhausted_is_incomplete():
    dets = chain_dets()
    r = solver.locate_conflict(
        dets, [{"a": 0, "b": 1, "type": "same"}], W_DIST, W_APP, PENALTY,
        total_time_limit=0.0,
    )
    assert r["status"] == "incomplete"


@pytest.mark.parametrize("seed", range(30))
def test_locate_on_random_infeasible_cases(seed):
    """随机实例：若定位成功，必须满足最小性保证。"""
    import random

    rng = random.Random(1000 + seed)
    n = rng.randint(3, 6)
    dets = [
        mk(f"b{k}", rng.randrange(4), rng.uniform(0, 100), rng.uniform(0, 100),
           [rng.uniform(0.1, 1) for _ in range(3)])
        for k in range(n)
    ]
    judgments = []
    for _ in range(rng.randint(1, 6)):
        a, b = rng.sample(range(n), 2)
        judgments.append({"a": a, "b": b, "type": rng.choice(["same", "different"])})
    r = solver.locate_conflict(dets, judgments, W_DIST, W_APP, PENALTY)
    if r["status"] == "found":
        check_minimal(dets, judgments, r["judgments"])
    elif r["status"] == "feasible":
        assert not is_infeasible(dets, judgments)
    else:
        assert r["status"] == "incomplete"
