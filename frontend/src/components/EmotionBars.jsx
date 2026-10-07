import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EMOTION_COLORS, label, sortedProbabilities } from "../utils/emotions";

// Seven-emotion probability bars for one face. Colour follows the emotion; table equivalent is in the aria-label.
export default function EmotionBars({ probabilities, height = 250 }) {
  const data = sortedProbabilities(probabilities).map((d) => ({ ...d, name: label(d.emotion) }));
  const summary = data.map((d) => `${d.name} ${(d.value * 100).toFixed(1)}%`).join(", ");
  return (
    <div role="img" aria-label={`Emotion probabilities: ${summary}`}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 52, bottom: 4, left: 4 }} barCategoryGap={6}>
          <XAxis type="number" domain={[0, 1]} hide />
          <YAxis type="category" dataKey="name" width={72} axisLine={false} tickLine={false} tick={{ fill: "#c3c2b7", fontSize: 12 }} />
          <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} formatter={(v) => `${(v * 100).toFixed(2)}%`} labelStyle={{ color: "#f2f1ec" }}
                   contentStyle={{ background: "#222220", border: "1px solid #2f2f2c", borderRadius: 8, fontSize: 12 }} />
          <Bar dataKey="value" barSize={14} radius={[0, 4, 4, 0]} background={{ fill: "rgba(255,255,255,0.04)", radius: 4 }}>
            {data.map((d) => <Cell key={d.emotion} fill={EMOTION_COLORS[d.emotion]} />)}
            <LabelList dataKey="value" position="right" formatter={(v) => `${(v * 100).toFixed(1)}%`} style={{ fill: "#c3c2b7", fontSize: 12 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
