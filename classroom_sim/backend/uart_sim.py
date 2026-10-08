"""
uart_sim.py

Preserves the real embedded-system packet format:

    ENG,<score>,<level>,<focused_count>,<device_count>,<sleep_count>\n

sent over UART0 (PA0 RX / PA1 TX) to the TM4C123GH6PM at 9600 8-N-1 in the
real system. In SIMULATION mode there is no COM port and no hardware --
packets are logged in-memory (and echoed to the server console tagged
"[SIMULATED UART]") instead of written to a serial port, so nothing ever
crashes for lack of a physical board.

If a real project already has a `pyserial`-based UART sender and a COM
port is actually present, that sender can be wired in behind the same
`send()` interface for LIVE mode -- this module only ever represents the
SIMULATION-mode path.
"""

from __future__ import annotations

from typing import List, Dict
import time


class SimulatedUART:
    def __init__(self, max_log: int = 200):
        self.last_packet: str = ""
        self._log: List[Dict] = []
        self._max_log = max_log

    def send(self, packet: str, simulation_time: float):
        self.last_packet = packet
        entry = {
            "simulation_time": round(simulation_time, 1),
            "packet": packet,
            "wall_time": time.strftime("%H:%M:%S"),
        }
        self._log.append(entry)
        if len(self._log) > self._max_log:
            self._log.pop(0)
        # Mirrors what the real UART sender would push to the wire; here it
        # just goes to stdout so it's visible during a live demo.
        print(f"[SIMULATED UART] {packet}")

    def recent(self, n: int = 20) -> List[Dict]:
        return self._log[-n:]
