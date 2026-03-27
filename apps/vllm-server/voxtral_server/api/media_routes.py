"""Media file endpoints — list, upload, delete."""

from fastapi import APIRouter, UploadFile

from ..media.manager import delete_media, list_media, save_upload
from ..models import ApiResponse

router = APIRouter()


@router.get("/api/media")
async def get_media():
    files = await list_media()
    return ApiResponse.ok([f.model_dump() for f in files])


@router.post("/api/media/upload")
async def upload_media(file: UploadFile):
    if not file.filename:
        return ApiResponse.err("No filename provided")
    data = await file.read()
    media_file = await save_upload(file.filename, data)
    return ApiResponse.ok(media_file.model_dump())


@router.delete("/api/media/{media_id}")
async def remove_media(media_id: str):
    if delete_media(media_id):
        return ApiResponse.ok({"deleted": media_id})
    return ApiResponse.err(f"Media '{media_id}' not found")
