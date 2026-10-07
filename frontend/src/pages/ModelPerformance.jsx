import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Disclaimer, EmptyState, ErrorBox, PageHeader, Spinner } from "../components/ui";
import { useApi } from "../hooks/usePolling";
import { api } from "../services/api";
import { EMOTIONS, label } from "../utils/emotions";

const KIND_COLOR = { classical: "#3987e5", control: "#8a8a85", hybrid: "#d4538c" };
const f = (v, d = 4) => (v?.mean == null ? "—" : v.mean.toFixed(d) + (v.n > 1 ? ` ± ${v.std.toFixed(d)}` : ""));

function Confusion({ matrix }) {
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead><tr><th className="p-1 text-muted">true ↓ / pred →</th>{EMOTIONS.map((e) => <th key={e} className="p-1 font-medium text-muted">{label(e)}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-1 pr-3 text-left font-medium text-muted">{label(EMOTIONS[i])}</th>
              {row.map((v, j) => {
                const frac = rowSums[i] ? v / rowSums[i] : 0;
                return <td key={j} title={`${v} (${(frac * 100).toFixed(1)}% of true ${EMOTIONS[i]})`} className="h-10 w-14 text-center tabular-nums text-ink"
                  style={{ background: `rgba(57,135,229,${0.08 + frac * 0.8})`, outline: i === j ? "1px solid #e8e8e4" : "none" }}>{v}</td>;
              })}
            </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPerformance() {
  const m = useApi(() => api.modelMetrics(), []);
  const d = m.data;
  const table = d?.comparison?.table || {};
  const kinds = ["classical", "control", "hybrid"].filter((k) => table[k]);
  const metricData = ["accuracy", "macro_precision", "macro_recall", "macro_f1"].map((key) => ({
    metric: { accuracy: "Accuracy", macro_precision: "Precision", macro_recall: "Recall", macro_f1: "F1" }[key],
    ...Object.fromEntries(kinds.map((k) => [k, table[k][key].mean])),
  }));
  const perClass = EMOTIONS.map((e) => ({ emotion: label(e), ...Object.fromEntries(kinds.map((k) => [k, table[k].per_class_f1[e].mean])) }));
  const [selKind] = [kinds.includes("hybrid") ? "hybrid" : kinds[0]];
  return (
    <>
      <PageHeader title="Model Performance" subtitle="Classical ResNet50 vs hybrid ResNet50 + quantum VQC, measured on the FER2013 test split. Only real evaluation results are shown." />
      <ErrorBox error={m.error} className="mb-4" />
      {!d && m.loading ? <Spinner /> : d && !d.available ? (
        <EmptyState title="No measured results yet">{d.message}</EmptyState>
      ) : d && (
        <div className="space-y-6">
          <Card title="Summary"><p className="text-sm leading-relaxed text-ink2">{d.comparison.summary}</p></Card>
          <Card title="Measured metrics" subtitle="mean ± std over seeds">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="text-left text-xs uppercase tracking-wider text-muted"><th className="py-2">Metric</th>{kinds.map((k) => <th key={k} className="py-2">{table[k].label}</th>)}</tr></thead>
                <tbody className="divide-y divide-line tabular-nums text-ink2">
                  {[["Accuracy", "accuracy"], ["Macro precision", "macro_precision"], ["Macro recall", "macro_recall"], ["Macro F1", "macro_f1"], ["Weighted F1", "weighted_f1"]].map(([n, k]) => (
                    <tr key={k}><td className="py-2 text-ink">{n}</td>{kinds.map((kk) => <td key={kk}>{f(table[kk][k])}</td>)}</tr>))}
                  <tr><td className="py-2 text-ink">Training time (s)</td>{kinds.map((k) => <td key={k}>{f(table[k].train_time_s, 1)}</td>)}</tr>
                  <tr><td className="py-2 text-ink">Head inference (ms/face)</td>{kinds.map((k) => <td key={k}>{f(table[k].head_ms_per_face, 3)}</td>)}</tr>
                  <tr><td className="py-2 text-ink">Total inference (ms/face)</td>{kinds.map((k) => <td key={k}>{f(table[k].total_ms_per_face, 2)}</td>)}</tr>
                  <tr><td className="py-2 text-ink">Head parameters</td>{kinds.map((k) => <td key={k}>{table[k].head_parameters.toLocaleString()} <span className="text-muted">({table[k].quantum_parameters} quantum)</span></td>)}</tr>
                  <tr><td className="py-2 text-ink">Seeds</td>{kinds.map((k) => <td key={k}>{table[k].seeds.join(", ")}</td>)}</tr>
                </tbody>
              </table>
            </div>
          </Card>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Overall metrics">
              <ResponsiveContainer width="100%" height={280}><BarChart data={metricData}><CartesianGrid stroke="#2c2c2a" vertical={false} /><XAxis dataKey="metric" stroke="#8a8a85" /><YAxis domain={[0, 1]} stroke="#8a8a85" /><Tooltip contentStyle={{ background: "#232321", border: "1px solid #3a3a37" }} /><Legend />{kinds.map((k) => <Bar key={k} dataKey={k} name={table[k].label} fill={KIND_COLOR[k]} radius={[4, 4, 0, 0]} />)}</BarChart></ResponsiveContainer>
            </Card>
            <Card title="Per-class F1">
              <ResponsiveContainer width="100%" height={280}><BarChart data={perClass}><CartesianGrid stroke="#2c2c2a" vertical={false} /><XAxis dataKey="emotion" stroke="#8a8a85" /><YAxis domain={[0, 1]} stroke="#8a8a85" /><Tooltip contentStyle={{ background: "#232321", border: "1px solid #3a3a37" }} /><Legend />{kinds.map((k) => <Bar key={k} dataKey={k} name={table[k].label} fill={KIND_COLOR[k]} radius={[4, 4, 0, 0]} />)}</BarChart></ResponsiveContainer>
            </Card>
          </div>
          {selKind && <Card title={`Confusion matrix — ${table[selKind].label}`} subtitle="Counts summed over seeds; cell shade = share of the true class"><Confusion matrix={table[selKind].confusion_matrix_sum} /></Card>}
          <Disclaimer>Quantum circuit is classically simulated. Results come from one dataset and a few seeds; they do not demonstrate quantum advantage or speed-up unless the statistics above say so.</Disclaimer>
        </div>
      )}
    </>
  );
}
