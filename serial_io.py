import time
import serial

from laser_protocol import compose_frame, default_checksum_hex_2


class SerialLineIO:
    """
    Robust serial line reader/writer.

    - Keeps leftover bytes between calls so we never lose data.
    - Uses the framed laser protocol only: send "$XXYYYYCC\\r\\n"
    """

    def __init__(
        self,
        ser: serial.Serial,
        *,
        log_fn,
        log_data_lines: bool = False,
        data_log_every: int = 50,
        default_command_id_hex2: str = "00",
        checksum_fn=default_checksum_hex_2,
    ):
        self.ser = ser
        self.log_fn = log_fn
        self.buf = b""
        self.log_data_lines = log_data_lines
        self.data_log_every = max(1, int(data_log_every))
        self._data_seen = 0
        self.default_command_id_hex2 = (default_command_id_hex2 or "00").strip().upper()
        self.checksum_fn = checksum_fn

    def write_command(self, data: str, *, command_id_hex2: str | None = None) -> None:
        """
        Writes a framed command: "$XXYYYYCC\\r\\n"

        - `data` is the YYYY payload (variable length, ASCII)
        - command id is XX (2 hex digits)
        """
        cid = self.default_command_id_hex2 if command_id_hex2 is None else str(command_id_hex2).strip().upper()
        framed = compose_frame(cid, data, checksum_fn=self.checksum_fn)
        self.log_fn(f"TX → {framed!r}")
        self.ser.write(framed)

    def read_line(self, timeout: float = 2.0) -> str:
        t0 = time.time()

        while time.time() - t0 < timeout:
            if b"\n" in self.buf:
                line_bytes, self.buf = self.buf.split(b"\n", 1)
                line = line_bytes.decode("ascii", errors="ignore").strip()

                # Logging policy: always log OK/ERR; optionally log some DATA
                if line.startswith("DATA"):
                    self._data_seen += 1
                    if self.log_data_lines and (self._data_seen % self.data_log_every == 0):
                        self.log_fn(f"RX ← {line}")
                else:
                    self.log_fn(f"RX ← {line}")

                return line

            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk
            else:
                time.sleep(0.01)

        raise TimeoutError("Timed out waiting for serial data")

