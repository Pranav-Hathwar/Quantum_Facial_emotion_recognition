import { Card, Disclaimer, EmptyState, ErrorBox, PageHeader, Spinner, StatTile } from "../components/ui";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";

const COL_W = 46, ROW_H = 46, PAD_L = 64, PAD_T = 28;
const GATE_FILL = { RY: "#3987e5", RX: "#199e70", RZ: "#c98a1a", M: "#8a8a85" };

function CircuitSvg({ spec }) {
  const n = spec.n_qubits;
  // lay out one gate column per rotation gate per stage, CNOT ring column after each layer
  const cols = [];
  spec.stages.forEach((st) => {
    if (st.stage === "encoding") st.gates.forEach(() => {});
    if (st.stage === "encoding") cols.push({ stage: "Encoding", gates: st.gates });
    else if (st.stage === "measurement") cols.push({ stage: "Measure", gates: st.gates });
    else {
      ["RX", "RY", "RZ"].forEach((g) => cols.push({ stage: st.stage.replace("variational_layer_", "Layer "), gates: st.gates.filter((x) => x.gate === g) }));
      st.entanglement?.forEach((k, i) => cols.push({ stage: i === 0 ? "CNOT ring" : "", cnots: [k] }));
    }
  });
  const W = PAD_L + cols.length * COL_W + 20, H = PAD_T + n * ROW_H + 10;
  const y = (w) => PAD_T + w * ROW_H + ROW_H / 2;
  return (
    <div className="overflow-x-auto">
      <svg width={W} height={H} role="img" aria-label="Variational quantum circuit diagram" className="min-w-full">
        {Array.from({ length: n }, (_, i) => (
          <g key={i}>
            <text x={8} y={y(i) + 4} fontSize="12" fill="#a8a8a2" fontFamily="monospace">q{i} |0⟩</text>
            <line x1={PAD_L} x2={W - 10} y1={y(i)} y2={y(i)} stroke="#3a3a37" strokeWidth="1.5" />
          </g>
        ))}
        {cols.map((c, ci) => {
          const x = PAD_L + ci * COL_W + COL_W / 2;
          return (
            <g key={ci}>
              <text x={x} y={14} fontSize="9" textAnchor="middle" fill="#8a8a85">{c.stage}</text>
              {c.gates?.map((g) => {
                const meas = g.gate.startsWith("expval");
                const fill = meas ? GATE_FILL.M : GATE_FILL[g.gate];
                return (
                  <g key={g.wire}>
                    <rect x={x - 17} y={y(g.wire) - 15} width="34" height="30" rx="5" fill={fill} stroke="#1a1a19" strokeWidth="2" />
                    <text x={x} y={y(g.wire) + 4} textAnchor="middle" fontSize={meas ? 10 : 11} fontWeight="600" fill="#fff">{meas ? "⟨Z⟩" : g.gate}</text>
                  </g>
                );
              })}
              {c.cnots?.map((k, ki) => {
                const off = (ki % 2) * 0; // all drawn in the same column; control dot + target ⊕
                return (
                  <g key={ki} opacity="0.85">
                    <line x1={x + off} x2={x + off} y1={y(k.control)} y2={y(k.target)} stroke="#a8a8a2" strokeWidth="1.2" strokeDasharray={Math.abs(k.control - k.target) > 1 ? "3 3" : ""} />
                    <circle cx={x} cy={y(k.control)} r="3.5" fill="#e8e8e4" />
                    <circle cx={x} cy={y(k.target)} r="7" fill="none" stroke="#e8e8e4" strokeWidth="1.5" />
                    <line x1={x - 7} x2={x + 7} y1={y(k.target)} y2={y(k.target)} stroke="#e8e8e4" strokeWidth="1.5" />
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Flow() {
  const steps = [
    ["ResNet50", "2048-d features", "classical"], ["Reduce", "Linear → n_qubits", "classical"],
    ["Encode", "R_y(π·tanh(x))", "quantum"], ["VQC", "RX/RY/RZ + CNOTs", "quantum"],
    ["Measure", "⟨Z⟩ per qubit", "quantum"], ["Fuse", "concat classical + quantum", "classical"], ["Classifier", "softmax · 7 emotions", "classical"],
  ];
  return (
    <ol className="flex flex-wrap items-stretch gap-2">
      {steps.map(([t, d, dom], i) => (
        <li key={t} className="flex items-center gap-2">
          <div className={`rounded-lg border px-3 py-2 ${dom === "quantum" ? "border-accent/60 bg-accent/10" : "border-line bg-surface2/60"}`}>
            <div className="text-sm font-semibold text-ink">{t}</div><div className="text-xs text-muted">{d}</div>
            <div className="mt-1 text-[10px] uppercase tracking-wider text-muted">{dom === "quantum" ? "quantum circuit (simulated)" : "classical"}</div>
          </div>
          {i < steps.length - 1 && <span className="text-muted" aria-hidden>→</span>}
        </li>
      ))}
    </ol>
  );
}

export default function QuantumModel() {
  const q = useApi(() => api.quantumCircuit(), []);
  const d = q.data;
  return (
    <>
      <PageHeader title="Quantum Circuit" subtitle="The variational quantum circuit inside the hybrid model (Chapter 12): angle encoding, trainable rotations with entanglement, Pauli-Z measurement." />
      <ErrorBox error={q.error} className="mb-4" />
      {!d && q.loading ? <Spinner /> : d && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile label="Qubits" value={d.n_qubits} />
            <StatTile label="Variational layers" value={d.n_layers} />
            <StatTile label="Trainable circuit params" value={d.n_trainable_parameters} />
            <StatTile label="Backend" value={<span className="text-base">default.qubit</span>} hint="classical simulator" />
          </div>
          <Card title="Pipeline" subtitle="Where the quantum circuit sits in the hybrid model"><Flow /></Card>
          <Card title="Circuit diagram" subtitle={`${d.encoding} · ${d.variational} · ${d.measurement}`}><CircuitSvg spec={d} /></Card>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Text diagram (PennyLane)"><pre className="overflow-x-auto text-xs leading-5 text-ink2">{d.diagram}</pre></Card>
            <Card title="Trained circuit weights" subtitle={d.trained ? `Run ${d.trained.run}` : undefined}>
              {d.trained ? (
                <pre className="max-h-72 overflow-auto text-xs text-ink2">{d.trained.weights.map((layer, l) => `Layer ${l + 1}\n` + layer.map((r, i) => `  q${i}: [${r.map((v) => v.toFixed(3)).join(", ")}]`).join("\n")).join("\n")}</pre>
              ) : <EmptyState title="No trained hybrid model loaded">Showing the architecture only. Weights appear once a hybrid checkpoint exists.</EmptyState>}
            </Card>
          </div>
          <Disclaimer>{d.note} Quantum components here are simulated on classical hardware; any accuracy difference versus the classical baseline is reported on the Model Performance page only if measured.</Disclaimer>
        </div>
      )}
    </>
  );
}
