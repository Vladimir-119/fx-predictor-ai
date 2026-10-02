"""Приветствие из шаблона преподавателя."""

import logging

from fastapi import APIRouter, Request

from src.schemas import HelloResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=HelloResponse)
async def hello(request: Request) -> HelloResponse:
    """Вернуть приветствие с названием и версией приложения."""
    settings = request.app.state.settings
    log.info("hello_requested")
    return HelloResponse(message="Hello, world!", app=settings.app_name, version=settings.version)
