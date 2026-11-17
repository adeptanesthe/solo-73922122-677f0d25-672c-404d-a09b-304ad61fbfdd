"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废。"""

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 冲突定位

def wait_locate(sid, timeout=15.0):
    """启动定位并等待结果。"""
    resp = client.post(f"/api/sessions/{sid}/locate-conflict")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if body.get("status") == "too_many":
        return body
    job = wait_job(body["job_id"], timeout=timeout)
    assert job["status"] == "done", job
    return job["result"]


def test_locate_conflict_returns_minimal_group():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    judgments = [
        {"a": "A0", "b": "B0", "type": "same"},   # 同帧同只，冲突
        {"a": "A1", "b": "A2", "type": "same"},   # 无关判断，不得混入
    ]
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": judgments})
    solved = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert solved["result"]["feasible"] is False
    assert solved["result"]["reason"] == "infeasible"

    res = wait_locate(sid)
    assert res["status"] == "found", res
    keys = {(j["a"], j["b"], j["type"]) for j in res["judgments"]}
    assert keys == {("A0", "B0", "same")}
    assert res["pair_keys"] == ["A0||B0"]

    # 删除该组后应恢复可行
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": []})
    ok = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert ok["result"]["feasible"] is True and ok["result"]["animal_count"] == 2


def test_locate_conflict_transitive_group_and_removal_guarantee():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    judgments = [
        {"a": "A0", "b": "A1", "type": "same"},
        {"a": "A1", "b": "A2", "type": "same"},
        {"a": "A0", "b": "A2", "type": "different"},
        {"a": "B1", "b": "B2", "type": "same"},
    ]
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": judgments})
    res = wait_locate(sid)
    assert res["status"] == "found", res
    group = res["judgments"]
    assert len(group) == 3, group
    assert all(g in judgments for g in group)

    # 组内任意一条删去后，单独用剩余判断必须是可行解
    for i in range(len(group)):
        trial = [j for j in group if j is not group[i]]
        client.put(f"/api/sessions/{sid}/judgments", json={"judgments": trial})
        done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
        assert done["result"]["feasible"] is True, (i, trial, done["result"])


def test_locate_conflict_over_limit_returns_hint_without_job():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    base = [{"a": "A0", "b": "A1", "type": "same"},
            {"a": "A1", "b": "A2", "type": "same"},
            {"a": "A2", "b": "A3", "type": "same"},
            {"a": "A3", "b": "A4", "type": "same"},
            {"a": "A4", "b": "A5", "type": "same"},
            {"a": "B0", "b": "B1", "type": "same"},
            {"a": "B1", "b": "B2", "type": "same"},
            {"a": "B2", "b": "B4", "type": "same"},
            {"a": "B4", "b": "B5", "type": "same"}]
    assert len(base) == 9
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": base})
    resp = client.post(f"/api/sessions/{sid}/locate-conflict")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "too_many" and body["job_id"] is None
    assert "8" in body["message"] and "仍可正常求解" in body["message"]

    # 超过上限仍可正常求解（这些判断本身不冲突）
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    assert done["status"] == "done"
    assert done["result"]["feasible"] is True
    assert done["result"]["animal_count"] == 2


def test_locate_conflict_feasible_input():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "A0", "b": "A1", "type": "same"}]})
    res = wait_locate(sid)
    assert res["status"] == "feasible"
    assert "judgments" not in res


def test_locate_conflict_job_version_tracks_input():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    resp = client.post(f"/api/sessions/{sid}/locate-conflict").json()
    assert resp["version"] == 2
    # 定位期间改判 → 任务仍是旧版本，前端据此丢弃
    client.put(f"/api/sessions/{sid}/judgments", json={"judgments": []})
    job = wait_job(resp["job_id"])
    assert job["version"] == 2
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3


def test_locate_conflict_unknown_session_404():
    assert client.post("/api/sessions/nope/locate-conflict").status_code == 404


def test_locate_conflict_solver_error_reports_incomplete(monkeypatch):
    """任一次可行性检查出错 → 明确提示定位未完成，不返回冲突组。"""
    import app.solver as solver_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated check failure")

    p = client.post("/api/example").json()
    sid = p["session_id"]
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    monkeypatch.setattr(solver_mod, "_run_milp", boom)
    res = wait_locate(sid)
    assert res["status"] == "incomplete"
    assert "judgments" not in res and "pair_keys" not in res
    assert "未完成" in res["message"]


def test_limits_expose_location_cap():
    p = make_session()
    assert p["limits"]["location_judgments"] == 8



def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}
