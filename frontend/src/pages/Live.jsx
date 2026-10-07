import { useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import EmotionBars from "../components/EmotionBars";
import FaceOverlay from "../components/FaceOverlay";
import { Button, Card, Disclaimer, ErrorBox, PageHeader, SeverityBadge, inputCls } from "../components/ui";
import { useSurveillanceSocket } from "../hooks/useSurveillanceSocket";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { EMOTION_COLORS, label, pct } from "../utils/emotions";

const INTERVALS = [{ ms: 500, text: "2 FPS (500 ms)" }, { ms: 1000, text: "1 FPS" }, { ms: 2000, text: "0.5 FPS" }];

export default function Live() {
  const cams = useApi(() => api.cameras(), []);
  const [cameraId, setCameraId] = useState("");
  const [source, setSource] = useState("webcam");
  const [intervalMs, setIntervalMs] = useState(500);
  const [running, setRunning] = useState(false);
  const [mediaError, setMediaError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => { if (!cameraId && cams.data?.length) setCameraId(cams.data[0].id); }, [cams.data, cameraId]);
  const sock = useSurveillanceSocket(cameraId, running);
  const canSend = sock.hello?.can_send_frames !== false;

  const stop = () => {
    setRunning(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) { videoRef.current.srcObject = null; if (source === "webcam") videoRef.current.removeAttribute("src"); }
  };
  const start = async () => {
    setMediaError(null);
    try {
      const v = videoRef.current;
      if (source === "webcam") {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser does not support camera access (needs HTTPS or localhost).");
        streamRef.current = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 540 } }, audio: false });
        v.srcObject = streamRef.current;
      } else {
        if (!videoUrl) throw new Error("Choose a video file first.");
        v.src = videoUrl; v.loop = true;
      }
      await v.play();
      setRunning(true);
    } catch (e) {
      const msg = e.name === "NotAllowedError" ? "Camera permission denied." : e.name === "NotFoundError" ? "No camera found on this device." : e.message;
      setMediaError(new Error(`Camera unavailable: ${msg}`));
    }
  };
  useEffect(() => () => { streamRef.current?.getTracks().forEach((t) => t.stop()); }, []);
  useEffect(() => () => videoUrl && URL.revokeObjectURL(videoUrl), [videoUrl]);

  // capture + send frames at the configured interval (never faster than the server can answer: busy frames are skipped)
  useEffect(() => {
    if (!running || sock.status !== "open" || !canSend) return undefined;
    const id = setInterval(() => {
      const v = videoRef.current, c = canvasRef.current;
      if (!v || !c || v.readyState < 2 || !v.videoWidth) return;
      const scale = Math.min(1, 640 / v.videoWidth);
      c.width = Math.round(v.videoWidth * scale); c.height = Math.round(v.videoHeight * scale);
      c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
      c.toBlob((b) => b && sock.sendFrame(b), "image/jpeg", 0.8);
    }, intervalMs);
    return () => clearInterval(id);
  }, [running, sock.status, canSend, intervalMs, sock.sendFrame]); // eslint-disable-line react-hooks/exhaustive-deps

  const frame = sock.frame;
  const face = useMemo(() => frame?.faces.find((f) => f.face_id === selected) || frame?.faces[0], [frame, selected]);
  const log = useMemo(() => [...sock.history].reverse().slice(0, 12), [sock.history]);

  return (
    <>
      <PageHeader title="Live Surveillance" subtitle="Browser camera or a video file feeds a simulated camera. Frames are analysed at a fixed rate; each face is predicted separately." />
      <div className="mb-4 flex flex-wrap items-end gap-3">
        <label className="text-xs text-muted">Camera
          <select className={`${inputCls} mt-1 !w-56`} value={cameraId} onChange={(e) => setCameraId(e.target.value)} disabled={running}>
            {(cams.data || []).map((c) => <option key={c.id} value={c.id}>{c.id} · {c.name}</option>)}
          </select></label>
        <label className="text-xs text-muted">Source
          <select className={`${inputCls} mt-1 !w-40`} value={source} onChange={(e) => setSource(e.target.value)} disabled={running}>
            <option value="webcam">Webcam</option><option value="file">Video file</option></select></label>
        {source === "file" && <input type="file" accept="video/*" className="text-xs text-ink2" disabled={running} onChange={(e) => e.target.files?.[0] && setVideoUrl(URL.createObjectURL(e.target.files[0]))} />}
        <label className="text-xs text-muted">Analysis rate
          <select className={`${inputCls} mt-1 !w-44`} value={intervalMs} onChange={(e) => setIntervalMs(+e.target.value)}>
            {INTERVALS.map((i) => <option key={i.ms} value={i.ms}>{i.text}</option>)}</select></label>
        {!running ? <Button onClick={start} disabled={!cameraId}>Start camera</Button> : <Button variant="danger" onClick={stop}>Stop</Button>}
        <span className="ml-auto text-xs text-muted">Socket: <span className={sock.status === "open" ? "text-ok" : "text-warn"}>{sock.status}</span></span>
      </div>
      <ErrorBox error={mediaError || sock.error} className="mb-4" />
      {running && !canSend && <ErrorBox error={new Error("Your role (Viewer) can watch events but cannot send camera frames. Ask an Operator or Admin.")} className="mb-4" />}
      {sock.alerts.filter((a) => a.status !== "RESOLVED").slice(0, 2).map((a) => (
        <div key={a.id} role="alert" className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-crit/50 bg-crit/10 px-4 py-3 text-sm">
          <span className="text-lg" aria-hidden="true">⚠</span><strong className="text-crit">EMOTIONAL ANOMALY DETECTED</strong><SeverityBadge severity={a.severity} />
          <span className="text-ink2">{a.camera_id} · score {a.anomaly_score.toFixed(2)} · avg {a.face_count.toFixed(1)} faces · for human review</span>
        </div>))}

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="space-y-6 xl:col-span-2">
          <Card title="LIVE CAMERA" subtitle={frame ? `${frame.face_count} face(s) · ${frame.timing_ms?.total_ms?.toFixed(0) ?? "—"} ms per frame · ${frame.model_kind} model` : "Not running"}>
            <FaceOverlay width={frame?.image_width || 0} height={frame?.image_height || 0} faces={frame?.faces} selected={face?.face_id} onSelect={setSelected}>
              <video ref={videoRef} muted playsInline className="block w-full min-h-[240px]" />
            </FaceOverlay>
            <canvas ref={canvasRef} className="hidden" />
            {frame?.warnings?.map((w) => <p key={w} className="mt-2 text-xs text-warn">{w}</p>)}
          </Card>
          <Card title="Distress-emotion share over time" subtitle="Share of faces showing Fear, Surprise or Angry in each processed frame. A single frame is not an alert; the engine needs a sustained, group-level shift from this camera's baseline.">
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={sock.history} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                <CartesianGrid stroke="#2f2f2c" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="time" stroke="#8f8e85" fontSize={11} tickLine={false} axisLine={false} minTickGap={40} />
                <YAxis stroke="#8f8e85" fontSize={11} tickLine={false} axisLine={false} unit="%" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#222220", border: "1px solid #2f2f2c", borderRadius: 8, fontSize: 12 }} formatter={(v) => `${v}%`} />
                <Line type="monotone" dataKey="distress" name="Distress share" stroke="#d95926" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
        <div className="space-y-6">
          <Card title="Faces in frame">
            {!frame?.faces.length ? <p className="text-sm text-muted">{running ? "No face in the current frame." : "Start the camera to begin."}</p> : (
              <ul className="space-y-2">
                {frame.faces.map((f) => (
                  <li key={f.face_id}>
                    <button onClick={() => setSelected(f.face_id)} className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left text-sm ${face?.face_id === f.face_id ? "border-accent bg-accent/10" : "border-line hover:bg-surface2"}`}>
                      <span className="text-muted">Face #{f.face_id}</span>
                      <span className="font-semibold" style={{ color: EMOTION_COLORS[f.emotion] }}>{label(f.emotion)}</span>
                      <span className="ml-auto tabular-nums text-ink2">{pct(f.confidence, 0)}</span>
                    </button>
                  </li>))}
              </ul>)}
          </Card>
          {face && <Card title={`Face #${face.face_id} probabilities`}><EmotionBars probabilities={face.probabilities} height={230} /></Card>}
          <Card title="Emotion timeline">
            {!log.length ? <p className="text-sm text-muted">No frames processed yet.</p> : (
              <ul className="space-y-1 font-mono text-xs">
                {log.map((h) => {
                  const top = Object.entries(h.counts).sort((a, b) => b[1] - a[1])[0];
                  return <li key={h.t} className="flex gap-2"><span className="text-muted">{h.time}</span><span className="text-ink2">→</span>
                    <span style={{ color: top ? EMOTION_COLORS[top[0]] : undefined }}>{top ? `${label(top[0])}${h.faces > 1 ? ` (${top[1]}/${h.faces})` : ""}` : "no face"}</span></li>;
                })}
              </ul>)}
          </Card>
        </div>
      </div>
      <div className="mt-6"><Disclaimer /></div>
    </>
  );
}
