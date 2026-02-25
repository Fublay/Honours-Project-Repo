import time
import serial

from protocol.frame_codec import compose_frame, default_checksum_hex_2
from protocol.command_composer import compose_set_pid_command
from protocol.reply_parser import parse_ack, parse_pid_reply


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
        cid = self.default_command_id_hex2 if command_id_hex2 is None else str(command_id_hex2).strip().upper()
        framed = compose_frame(cid, data, checksum_fn=self.checksum_fn)
        self.log_fn(f"TX -> {framed!r}")
        self.ser.write(framed)

    def read_line(self, timeout: float = 2.0, keep_terminator: bool = False) -> str:
        t0 = time.time()

        while time.time() - t0 < timeout:
            if b"\n" in self.buf:
                line_bytes, self.buf = self.buf.split(b"\n", 1)
                raw_line = (line_bytes + b"\n").decode("ascii", errors="ignore")
                line = raw_line.rstrip("\r\n")

                if line.startswith("DATA"):
                    self._data_seen += 1
                    if self.log_data_lines and (self._data_seen % self.data_log_every == 0):
                        self.log_fn(f"RX <- {line}")
                else:
                    self.log_fn(f"RX <- {line}")

                if keep_terminator:
                    return raw_line
                return line

            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk
            else:
                time.sleep(0.01)

        raise TimeoutError("Timed out waiting for serial data")

    def get_pid_values(self, timeout: float = 2.0) -> dict:
        cmd_bytes = compose_frame("B6", "", checksum_fn=self.checksum_fn)
        self.log_fn(f"TX -> {cmd_bytes!r}")
        self.ser.write(cmd_bytes)
        deadline = time.time() + timeout
        last_error = None

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            reply = self.read_line(timeout=remaining, keep_terminator=True)
            try:
                return parse_pid_reply(reply)
            except Exception as e:
                last_error = e
                # Ignore unrelated frames/lines (e.g. debug stream) while waiting for $B6 reply.
                continue

        if last_error is not None:
            raise ValueError(f"Failed to parse GET_PID reply before timeout: {last_error}")
        raise TimeoutError("Timed out waiting for GET_PID reply")

    def write_command_expect_ok_ack(
        self,
        data: str,
        *,
        command_id_hex2: str | None = None,
        timeout: float = 2.0,
    ) -> str:
        """
        Send command and wait for success ACK '*00'.
        Ignores non-ACK lines (e.g. telemetry/debug) until timeout.
        """
        self.write_command(data, command_id_hex2=command_id_hex2)
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            line = self.read_line(timeout=remaining)
            ok_ack, _ = parse_ack(line)
            if not line.startswith("*"):
                continue
            if ok_ack:
                return line
            raise RuntimeError(f"Command failed with ACK: {line}")

        raise TimeoutError("Timed out waiting for success ACK '*00'")

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
        if current_values is None:
            try:
                current_values = self.get_pid_values(timeout=timeout)
            except Exception:
                current_values = None

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

        self.log_fn(f"TX -> {cmd_bytes!r}")
        self.ser.write(cmd_bytes)
        ack = self.read_line(timeout=timeout)
        return ack
