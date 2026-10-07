import { EMOTION_COLORS, label, pct } from "../utils/emotions";

// Draws bounding boxes + labels over an <img>/<video> child. Coordinates are in source-pixel space (width x height).
export default function FaceOverlay({ width, height, faces = [], selected, onSelect, children }) {
  const stroke = Math.max(2, Math.round(width / 220));
  const font = Math.max(12, Math.round(width / 38));
  return (
    <div className="relative w-full overflow-hidden rounded-lg bg-black">
      {children}
      {width > 0 && (
        <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
          {faces.map((f) => {
            const b = f.bounding_box;
            const color = EMOTION_COLORS[f.emotion] || "#fff";
            const text = `#${f.face_id} ${label(f.emotion)} ${pct(f.confidence, 0)}`;
            const w = text.length * font * 0.6 + font;
            const y = b.y - font * 1.5 < 0 ? b.y + b.height : b.y - font * 1.5;
            return (
              <g key={f.face_id} onClick={() => onSelect?.(f.face_id)} style={{ cursor: onSelect ? "pointer" : "default" }}>
                <rect x={b.x} y={b.y} width={b.width} height={b.height} fill="none" stroke={color} strokeWidth={selected === f.face_id ? stroke * 2 : stroke} rx={stroke} />
                <rect x={b.x} y={y} width={w} height={font * 1.4} fill={color} rx={stroke} />
                <text x={b.x + font * 0.4} y={y + font * 1.05} fontSize={font} fontWeight="600" fill="#0d0d0c" fontFamily="system-ui, sans-serif">{text}</text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
