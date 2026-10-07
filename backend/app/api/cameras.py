from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import Principal, current_user, get_db, require_role
from backend.app.models import Camera, Role
from backend.app.schemas.api import CameraCreate, CameraOut, CameraUpdate
from backend.app.services import store

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraOut])
def list_cameras(_: Principal = Depends(current_user), db: Session = Depends(get_db)):
    return [store.camera_out(db, c) for c in db.scalars(select(Camera).order_by(Camera.id)).all()]


@router.post("", response_model=CameraOut, status_code=201)
def create_camera(body: CameraCreate, _: Principal = Depends(require_role(Role.ADMIN)), db: Session = Depends(get_db)):
    cid = (body.id or store.next_camera_id(db)).strip().upper()
    if db.get(Camera, cid):
        raise HTTPException(409, f"Camera {cid} already exists.")
    cam = Camera(id=cid, name=body.name, location=body.location, source=body.source, status="OFFLINE")
    db.add(cam)
    db.commit()
    return store.camera_out(db, cam)


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, body: CameraUpdate, _: Principal = Depends(require_role(Role.ADMIN)),
                  db: Session = Depends(get_db)):
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(404, "Camera not found.")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(cam, k, v)
    db.commit()
    return store.camera_out(db, cam)
