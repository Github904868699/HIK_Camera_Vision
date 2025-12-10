"""Communication helpers for PLC/Modbus integration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from pymodbus.client import ModbusTcpClient  # type: ignore
except Exception:  # pragma: no cover
    ModbusTcpClient = None  # type: ignore


@dataclass
class ModbusConfig:
    host: str = "127.0.0.1"
    port: int = 502
    unit: int = 1


class ModbusBridge:
    def __init__(self, config: ModbusConfig) -> None:
        self.config = config
        self._client: Optional[ModbusTcpClient] = None

    def connect(self) -> None:
        if ModbusTcpClient is None:  # pragma: no cover - runtime guard
            raise RuntimeError("pymodbus not installed")
        self._client = ModbusTcpClient(self.config.host, port=self.config.port)
        self._client.connect()
        LOGGER.info("Connected to Modbus %s:%s", self.config.host, self.config.port)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def write_register(self, address: int, value: int) -> bool:
        if self._client is None:
            raise RuntimeError("Client not connected")
        result = self._client.write_register(address, value, unit=self.config.unit)
        return not getattr(result, "isError", lambda: True)()

    def read_register(self, address: int) -> int:
        if self._client is None:
            raise RuntimeError("Client not connected")
        result = self._client.read_holding_registers(address, 1, unit=self.config.unit)
        if getattr(result, "isError", lambda: True)():
            raise RuntimeError("Read failed")
        return int(result.registers[0])


__all__ = ["ModbusConfig", "ModbusBridge"]
