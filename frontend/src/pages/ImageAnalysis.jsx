import { useEffect, useMemo, useRef, useState } from "react";
import EmotionBars from "../components/EmotionBars";
import FaceOverlay from "../components/FaceOverlay";
import PipelineStatus from "../components/PipelineStatus";
import { Button, Card, Disclaimer, ErrorBox, PageHeader, Spinner, inputCls } from "../components/ui";
import { api } from "../services/api";
import { EMOTION_COLORS, label, pct } from "../utils/emotions";

export default function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState("");
  const [selected, setSelected] = useState(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const choose = (f) => {
    if (!f) return;
    setFile(f); setResult(null); setError(null); setSelected(null);
    setPreviewUrl(URL.createObjectURL(f));
  };
  const analyse = async () => {
    if (!file) return;
    setBusy(true); setError(null); setResult(null);
    try {
      const r = await api.predictMultiple(file, model || undefined);
      setResult(r);
      setSelected(r.faces[0]?.face_id ?? null);
    } catch (e) { setError(e); } finally { setBusy(false); }
  };
  const face = useMemo(() => result?.faces.find((f) => f.face_id === selected), [result, selected]);

  return (
    <>
      <PageHeader title="Image Analysis" subtitle="Upload a photo. Every detected face is analysed separately through the ResNet50 → quantum circuit → classifier pipeline." />
      <div className="grid gap-6 xl:grid-cols-5">
        <div className="space-y-6 xl:col-span-3">
          <Card title="Input image" actions={
            <div className="flex items-center gap-2">
              <select className={`${inputCls} !w-auto`} value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model">
                <option value="">Default model</option><option value="hybrid">Hybrid (ResNet50 + VQC)</option><option value="classical">Classical (ResNet50)</option>
              </select>
              <Button onClick={analyse} disabled={!file || busy}>{busy ? "Analysing…" : "Analyse"}</Button>
            </div>}>
            {!previewUrl ? (
              <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
                   onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files?.[0]); }}
                   onClick={() => inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
                   className={`flex h-72 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed text-center transition ${drag ? "border-accent bg-accent/5" : "border-line hover:border-muted"}`}>
                <p className="text-sm font-medium text-ink2">Drop an image here or click to browse</p>
                <p className="mt-1 text-xs text-muted">JPG, JPEG, PNG or WEBP · up to 10 MB</p>
              </div>
            ) : (
              <FaceOverlay width={result?.image_width || 0} height={result?.image_height || 0} faces={result?.faces} selected={selected} onSelect={setSelected}>
                <img src={previewUrl} alt="Uploaded for analysis" className="block w-full" />
              </FaceOverlay>
            )}
            <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,image/*" className="hidden" onChange={(e) => choose(e.target.files?.[0])} />
            {previewUrl && <button className="mt-3 text-xs text-accent hover:underline" onClick={() => inputRef.current?.click()}>Choose a different image</button>}
            <ErrorBox error={error} className="mt-4" />
            {busy && <Spinner label="Detecting faces and running the model…" />}
          </Card>
          <Disclaimer />
        </div>

        <div className="space-y-6 xl:col-span-2">
          {!result && !busy && <Card title="Result"><p className="text-sm text-muted">Results appear here: a prediction and all seven class probabilities for each face.</p></Card>}
          {result && (
            <>
              <Card title={`${result.face_count} face${result.face_count === 1 ? "" : "s"} detected`} subtitle={`Model: ${result.model_kind} · ${result.model}`}>
                <div className="flex flex-wrap gap-2">
                  {result.faces.map((f) => (
                    <button key={f.face_id} onClick={() => setSelected(f.face_id)}
                      className={`rounded-lg border px-3 py-1.5 text-left text-xs transition ${selected === f.face_id ? "border-accent bg-accent/10" : "border-line hover:bg-surface2"}`}>
                      <span className="text-muted">Face {f.face_id}</span><br />
                      <span className="font-semibold" style={{ color: EMOTION_COLORS[f.emotion] }}>{label(f.emotion)}</span> <span className="text-ink2">{pct(f.confidence)}</span>
                    </button>
                  ))}
                </div>
                {result.warnings.map((w) => <p key={w} className="mt-3 text-xs text-warn">{w}</p>)}
              </Card>
              {face && (
                <Card title={`Face #${face.face_id}`} subtitle="Estimated facial expression (probabilistic)">
                  <div className="mb-3 flex items-baseline gap-3">
                    <span className="text-2xl font-semibold uppercase tracking-wide" style={{ color: EMOTION_COLORS[face.emotion] }}>{face.emotion}</span>
                    <span className="text-lg tabular-nums text-ink2">{pct(face.confidence)}</span>
                  </div>
                  <EmotionBars probabilities={face.probabilities} />
                  {face.quantum_features && (
                    <div className="mt-4 rounded-lg border border-line bg-bg p-3 text-xs">
                      <div className="mb-1 font-semibold text-ink2">Quantum measurement (Pauli-Z expectation values)</div>
                      <div className="font-mono text-muted">[{face.quantum_features.map((v) => v.toFixed(3)).join(", ")}]</div>
                      <div className="mt-1 text-muted">Encoded angles: [{face.quantum_angles.map((v) => v.toFixed(2)).join(", ")}] rad</div>
                    </div>
                  )}
                </Card>
              )}
              <Card title="Pipeline status"><PipelineStatus stages={result.pipeline} timing={result.timing_ms} /></Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
