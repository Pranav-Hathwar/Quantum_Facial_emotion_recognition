// Fixed emotion order (FER2013 coding) and colours. Colours are validated for dark-surface categorical use;
// they follow the emotion, never its rank. The server also exposes them at /api/emotions.
export const EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];

export const EMOTION_COLORS = {
  angry: "#e66767", disgust: "#008300", fear: "#9085e9", happy: "#c98500",
  sad: "#3987e5", surprise: "#d95926", neutral: "#199e70",
};

export const label = (e) => (e ? e.charAt(0).toUpperCase() + e.slice(1) : "—");
export const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

// Emotions the anomaly engine treats as the "group distress" signal (never as a per-person verdict)
export const DISTRESS = ["fear", "surprise", "angry"];

export const SEVERITY_STYLE = {
  LOW: "bg-warn/15 text-warn border-warn/40",
  MEDIUM: "bg-serious/15 text-serious border-serious/40",
  HIGH: "bg-crit/15 text-crit border-crit/40",
  CRITICAL: "bg-crit text-white border-crit",
};

export function sortedProbabilities(probs) {
  return EMOTIONS.map((e) => ({ emotion: e, value: probs?.[e] ?? 0 })).sort((a, b) => b.value - a.value);
}
