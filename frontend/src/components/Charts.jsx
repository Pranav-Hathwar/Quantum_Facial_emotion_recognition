import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EMOTIONS, EMOTION_COLORS, label } from "../utils/emotions";

const tip = { contentStyle: { background: "#222220", border: "1px solid #2f2f2c", borderRadius: 8, fontSize: 12 }, labelStyle: { color: "#f2f1ec" }, itemStyle: { color: "#c3c2b7" } };
const axis = { stroke: "#8f8e85", fontSize: 11, tickLine: false, axisLine: false };

export function DistributionDonut({ distribution, height = 260 }) {
  const data = EMOTIONS.map((e) => ({ emotion: e, name: label(e), value: distribution?.[e]?.count ?? 0, percent: distribution?.[e]?.percent ?? 0 }));
  const total = data.reduce((a, d) => a + d.value, 0);
  if (!total) return <p className="py-10 text-center text-sm text-muted">No faces analysed in this time window yet.</p>;
  const summary = data.map((d) => `${d.name} ${d.percent}%`).join(", ");
  return (
    <div role="img" aria-label={`Emotion distribution: ${summary}`}>
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie data={data.filter((d) => d.value > 0)} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="88%" paddingAngle={2} stroke="#1a1a19" strokeWidth={2}>
            {data.filter((d) => d.value > 0).map((d) => <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />)}
          </Pie>
          <Tooltip {...tip} formatter={(v, n, p) => [`${v} faces (${p.payload.percent}%)`, n]} />
          <Legend iconType="circle" iconSize={8} formatter={(v) => <span className="text-xs text-ink2">{v}</span>} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DistributionBars({ distribution, height = 260 }) {
  const data = EMOTIONS.map((e) => ({ emotion: e, name: label(e), percent: distribution?.[e]?.percent ?? 0, count: distribution?.[e]?.count ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="#2f2f2c" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" {...axis} />
        <YAxis {...axis} unit="%" />
        <Tooltip {...tip} cursor={{ fill: "rgba(255,255,255,0.04)" }} formatter={(v, n, p) => [`${v}% (${p.payload.count} faces)`, "Share"]} />
        <Bar dataKey="percent" barSize={26} radius={[4, 4, 0, 0]}>
          {data.map((d) => <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// Stacked area of faces per emotion over time buckets
export function EmotionTimeline({ timeline, height = 280 }) {
  const data = (timeline || []).map((b) => ({ time: new Date(b.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), ...b.counts }));
  if (!data.length) return <p className="py-10 text-center text-sm text-muted">No timeline data yet. Start a live camera or analyse images.</p>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="#2f2f2c" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="time" {...axis} minTickGap={24} />
        <YAxis {...axis} allowDecimals={false} />
        <Tooltip {...tip} />
        <Legend iconType="circle" iconSize={8} formatter={(v) => <span className="text-xs text-ink2">{label(v)}</span>} />
        {EMOTIONS.map((e) => (
          <Area key={e} type="monotone" dataKey={e} name={e} stackId="1" stroke={EMOTION_COLORS[e]} strokeWidth={1.5} fill={EMOTION_COLORS[e]} fillOpacity={0.55} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

// Baseline vs current share of each emotion (for alert details)
export function BaselineCompare({ current, baseline, height = 240 }) {
  const data = EMOTIONS.map((e) => ({ name: label(e), Baseline: +(100 * (baseline?.[e] ?? 0)).toFixed(1), "During alert": +(100 * (current?.[e] ?? 0)).toFixed(1) }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2}>
        <CartesianGrid stroke="#2f2f2c" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" {...axis} />
        <YAxis {...axis} unit="%" />
        <Tooltip {...tip} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
        <Legend iconSize={10} formatter={(v) => <span className="text-xs text-ink2">{v}</span>} />
        <Bar dataKey="Baseline" fill="#5b5a54" barSize={12} radius={[3, 3, 0, 0]} />
        <Bar dataKey="During alert" fill="#3987e5" barSize={12} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
