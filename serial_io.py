import time
import serial

from laser_protocol import (
    compose_frame,
    default_checksum_hex_2,
    compose_set_pid_command,
    parse_pid_reply,
)
import laser_command_ids as CMD


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

    def get_pid_values(self, timeout: float = 2.0) -> dict:
        """
        Read current PID values from the laser controller.
        
        Returns:
            Dictionary with keys: pw_kp, pw_ki, pw_kd, pp_kp, pp_ki, pp_kd, holdoff, sample_interval
        
        Raises:
            TimeoutError if no reply received
            ValueError if reply format is invalid
        """
        # Send GET_PID command: $B600\r\n (no data, checksum is 00)
        cmd_bytes = compose_frame("B6", "", checksum_fn=self.checksum_fn)
        self.log_fn(f"TX → {cmd_bytes!r}")
        self.ser.write(cmd_bytes)
        
        # Read reply
        reply = self.read_line(timeout=timeout)
        
        # Parse PID values from reply
        return parse_pid_reply(reply)

    def set_pid_values(
        self,
        pw_kp: float,
        pw_ki: float,
        pw_kd: float,
        pp_kp: float | None = None,
        pp_ki: float | None = None,
        pp_kd: float | None = None,
        holdoff: float | None = None,
        sample_interval: float | None = None,
        current_values: dict | None = None,
        timeout: float = 2.0,
    ) -> str:
        """
        Set PID values on the laser controller.
        
        Args:
            pw_kp, pw_ki, pw_kd: Pulse Width PID parameters (required)
            pp_kp, pp_ki, pp_kd: Pulse Period PID parameters (optional)
            holdoff: Holdoff period in ms (optional)
            sample_interval: Sample interval in ms (optional)
            current_values: Current PID values dict (used for unspecified parameters)
            timeout: Timeout for reading acknowledgment
        
        Returns:
            Acknowledgment string (e.g., "*00\r\n")
        
        Raises:
            TimeoutError if no reply received
        """
        # If current_values not provided, read them first
        if current_values is None:
            try:
                current_values = self.get_pid_values(timeout=timeout)
            except Exception:
                # If we can't read current values, use defaults
                current_values = None
        
        # Compose SET_PID command
        cmd_bytes = compose_set_pid_command(
            pw_kp=pw_kp,
            pw_ki=pw_ki,
            pw_kd=pw_kd,
            pp_kp=pp_kp,
            pp_ki=pp_ki,
            pp_kd=pp_kd,
            holdoff=holdoff,
            sample_interval=sample_interval,
            current_values=current_values,
            checksum_fn=self.checksum_fn,
        )
        
        self.log_fn(f"TX → {cmd_bytes!r}")
        self.ser.write(cmd_bytes)
        
        # Read acknowledgment (format: *00\r\n)
        ack = self.read_line(timeout=timeout)
        return ack

