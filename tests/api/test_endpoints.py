import pytest
from httpx import AsyncClient
import os

test_params = {
    'pickup_datetime': '2013-07-06 10:18:00',
    'pickup_longitude': '-73.70',
    'pickup_latitude': '40.9',
    'dropoff_longitude': '-73.98',
    'dropoff_latitude': '40.70',
    'passenger_count': '2'
}

@pytest.mark.asyncio
async def test_root_is_up():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac: # La versión de httpx: Si la versión instalada de httpx no soporta app=app como argumento en make test_api_root solucion: pip install  "httpx<0.28"
        response = await ac.get("/")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_root_returns_greeting():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"greeting": "Hello"}


@pytest.mark.asyncio
async def test_predict_is_up():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_is_dict():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json(), dict)
    assert len(response.json()) == 1


@pytest.mark.asyncio
async def test_predict_has_key():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert response.json().get('fare', False)


@pytest.mark.asyncio
async def test_predict_val_is_float():
    from taxifare.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict", params=test_params)
    assert isinstance(response.json().get('fare'), float)
