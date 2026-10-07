import random

from backend.app.services.anomaly import AnomalyConfig, AnomalyDetector, DISTRESS_EMOTIONS


def _calm_frame(rng, n=4):
    return [rng.choices(["neutral", "happy", "sad"], [0.7, 0.25, 0.05])[0] for _ in range(n)]


def _panic_frame(rng, n=6, p=0.75):
    return [rng.choice(DISTRESS_EMOTIONS) if rng.random() < p else "neutral" for _ in range(n)]


def _run(det, cam, start, seconds, frame_fn, fps=2):
    ts, results = start, []
    for _ in range(int(seconds * fps)):
        results.append(det.observe(cam, ts, frame_fn()))
        ts += 1 / fps
    return ts, results


def test_calm_scene_never_alerts():
    rng, det = random.Random(1), AnomalyDetector()
    _, res = _run(det, "c", 0, 120, lambda: _calm_frame(rng))
    assert not any(r.is_anomalous for r in res)


def test_sustained_group_distress_after_calm_baseline_alerts():
    rng, det = random.Random(2), AnomalyDetector()
    t, _ = _run(det, "c", 0, 90, lambda: _calm_frame(rng))
    _, res = _run(det, "c", t, 25, lambda: _panic_frame(rng))
    assert res[-1].is_anomalous
    assert res[-1].severity in {"MEDIUM", "HIGH", "CRITICAL"}
    assert res[-1].score > 0.4 and res[-1].window_distribution["fear"] > 0.1


def test_single_angry_face_is_not_an_alert():
    rng, det = random.Random(3), AnomalyDetector()
    t, _ = _run(det, "c", 0, 90, lambda: _calm_frame(rng))
    _, res = _run(det, "c", t, 40, lambda: ["angry"] + _calm_frame(rng, 5))   # one angry face among 6
    assert not any(r.is_anomalous for r in res)


def test_single_person_fear_is_not_group_level():
    rng, det = random.Random(4), AnomalyDetector()
    t, _ = _run(det, "c", 0, 90, lambda: _calm_frame(rng))
    _, res = _run(det, "c", t, 40, lambda: ["fear"])           # one face, fully fearful, sustained
    assert not any(r.is_anomalous for r in res)
    assert "group-level" in res[-1].reason


def test_short_burst_is_ignored():
    rng, det = random.Random(5), AnomalyDetector()
    t, _ = _run(det, "c", 0, 90, lambda: _calm_frame(rng))
    t, burst = _run(det, "c", t, 4, lambda: _panic_frame(rng))
    assert not any(r.is_anomalous for r in burst)


def test_no_alert_before_baseline_is_trusted():
    rng, det = random.Random(6), AnomalyDetector(AnomalyConfig(baseline_warmup=10_000))
    _, res = _run(det, "c", 0, 40, lambda: _panic_frame(rng))
    assert not any(r.is_anomalous for r in res) and not res[-1].baseline_trusted


def test_baseline_does_not_absorb_the_anomaly_and_cameras_are_independent():
    rng, det = random.Random(7), AnomalyDetector()
    t, _ = _run(det, "A", 0, 90, lambda: _calm_frame(rng))
    _run(det, "A", t, 60, lambda: _panic_frame(rng))
    base = det.baseline("A")
    assert base[[0, 2, 5]].sum() < 0.25          # distress share of the baseline stayed low
    assert det.baseline("B").sum() > 0.99        # unrelated camera still on the prior
