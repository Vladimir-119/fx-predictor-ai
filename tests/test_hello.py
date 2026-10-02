"""Сохранённая главная страница шаблона."""


async def test_hello_returns_message_and_package_metadata(client, settings):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Hello, world!",
        "app": "fx-predictor-ai",
        "version": settings.version,
    }
