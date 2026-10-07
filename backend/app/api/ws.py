"""WebSocket: /ws/surveillance/{camera_id}  (binary JPEG frames in, JSON events out; other clients can just listen)."""
from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

from backend.app.api.deps import ROLE_RANK, principal_from_token
from backend.app.models import Role
from backend.app.services.surveillance import process_frame
from backend.app.utils.logging import get_logger
from ml.exceptions import InvalidImageError, QuantumVisionError
from ml.preprocessing.image_preprocessor import decode_image

router = APIRouter(tags=["realtime"])
log = get_logger("ws")


class ConnectionManager:
    def __init__(self):
        self._conns: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, camera_id: str, ws: WebSocket) -> None:
        self._conns[camera_id].add(ws)

    def disconnect(self, camera_id: str, ws: WebSocket) -> None:
        self._conns[camera_id].discard(ws)

    def count(self, camera_id: str) -> int:
        return len(self._conns[camera_id])

    async def broadcast(self, camera_id: str, event: dict) -> None:
        dead = []
        for ws in list(self._conns[camera_id]):
            try:
                await ws.send_json(event)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(camera_id, ws)


@router.websocket("/ws/surveillance/{camera_id}")
async def surveillance_socket(ws: WebSocket, camera_id: str, token: str | None = Query(None)):
    state = ws.app.state
    db = state.db.session()
    try:
        try:
            user = principal_from_token(state, db, token)
        except HTTPException:
            await ws.close(code=4401)
            return
        await ws.accept()
        await state.ws.connect(camera_id, ws)
        can_send = ROLE_RANK.get(user.role, 0) >= ROLE_RANK[Role.OPERATOR.value]
        await ws.send_json({"type": "connected", "camera_id": camera_id, "role": user.role, "can_send_frames": can_send,
                            "process_interval_ms": state.settings.process_interval_ms})
        busy = asyncio.Lock()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") == "ping":
                await ws.send_json({"type": "pong"})
                continue
            data = msg.get("bytes")
            if not data:
                continue
            if not can_send:
                await ws.send_json({"type": "error", "message": "Viewer role cannot send frames."})
                continue
            if busy.locked():                      # drop frames instead of queueing: the model is the bottleneck
                await ws.send_json({"type": "frame_dropped"})
                continue
            async with busy:
                try:
                    img = decode_image(data)
                    outcome = await run_in_threadpool(process_frame, state, db, img, camera_id, None)
                    await state.ws.broadcast(camera_id, {"type": "frame_result", "camera_id": camera_id,
                                                         **outcome.response.model_dump(mode="json", exclude={"alert"})})
                    for ev in outcome.events:
                        await state.ws.broadcast(camera_id, ev)
                except InvalidImageError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except LookupError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except QuantumVisionError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    log.error("frame processing failed", extra={"error": str(exc)[:200]})
                    db.rollback()
                    await ws.send_json({"type": "error", "message": "Frame processing failed."})
    except WebSocketDisconnect:
        pass
    finally:
        state.ws.disconnect(camera_id, ws)
        db.close()
