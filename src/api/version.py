"""Версия установленного приложения."""

from fastapi import APIRouter, Request

from src.schemas import VersionResponse

router = APIRouter()


@router.get("/api/v1/version", response_model=VersionResponse)
async def version(request: Request) -> VersionResponse:
    """Вернуть версию сборки; v1 в пути означает версию контракта API."""
    settings = request.app.state.settings
    return VersionResponse(app=settings.app_name, version=settings.version)
