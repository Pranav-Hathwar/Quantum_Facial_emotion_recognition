import { useCallback, useEffect, useRef, useState } from "react";
import { surveillanceSocketUrl } from "../services/api";

// Connects to /ws/surveillance/{camera}; exposes the latest frame result, an event log and active alerts.
export function useSurveillanceSocket(cameraId, enabled) {
  const wsRef = useRef(null);
  const busyRef = useRef(false);
  const [status, setStatus] = useState("idle");
  const [hello, setHello] = useState(null);
  const [frame, setFrame] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]); // [{t, faces, counts, distress}] one entry per processed frame

  useEffect(() => {
    if (!enabled || !cameraId) return undefined;
    setStatus("connecting"); setError(null); setHistory([]); setAlerts([]);
    const ws = new WebSocket(surveillanceSocketUrl(cameraId));
    wsRef.current = ws;
    ws.onopen = () => setStatus("open");
    ws.onclose = (e) => { setStatus(e.code === 4401 ? "unauthorized" : "closed"); busyRef.current = false; };
    ws.onerror = () => setError(new Error("WebSocket connection failed. Is the backend running?"));
    ws.onmessage = (m) => {
      let ev; try { ev = JSON.parse(m.data); } catch { return; }
      if (ev.type === "connected") setHello(ev);
      else if (ev.type === "frame_result") {
        busyRef.current = false;
        setFrame(ev);
        const counts = {};
        ev.faces.forEach((f) => { counts[f.emotion] = (counts[f.emotion] || 0) + 1; });
        const n = ev.faces.length;
        const distress = n ? ((counts.fear || 0) + (counts.surprise || 0) + (counts.angry || 0)) / n : 0;
        setHistory((h) => [...h.slice(-59), { t: Date.now(), time: new Date().toLocaleTimeString(), faces: n, counts, distress: +(distress * 100).toFixed(1) }]);
      } else if (ev.type === "alert_created" || ev.type === "alert_updated") {
        setAlerts((a) => [ev.alert, ...a.filter((x) => x.id !== ev.alert.id)].slice(0, 10));
      } else if (ev.type === "error") { busyRef.current = false; setError(new Error(ev.message)); }
      else if (ev.type === "frame_dropped") busyRef.current = false;
    };
    return () => { ws.close(); wsRef.current = null; setStatus("idle"); };
  }, [cameraId, enabled]);

  const sendFrame = useCallback((blob) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || busyRef.current) return false;
    busyRef.current = true;
    ws.send(blob);
    return true;
  }, []);

  return { status, hello, frame, alerts, error, history, sendFrame };
}
