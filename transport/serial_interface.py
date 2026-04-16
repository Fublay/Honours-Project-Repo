"""Serial transport implementation used by the PID tuner.

`SerialLineIO` is the boundary between business logic and the pyserial device.
It knows how to frame commands, read full lines, and wait for ACK/PID replies.
"""

# Everything above this layer should be able to think in terms of commands and
# parsed replies instead of raw serial bytes.

import time
import serial

from protocol.frame_codec import compose_frame, default_checksum_hex_2
from protocol.command_composer import compose_set_pid_command, compose_set_program_command
from protocol.reply_parser import parse_ack, parse_pid_reply, parse_program_reply


# Keep serial edge cases in one class so the rest of the code can stay mostly
# deterministic.
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

    # Most callers go through this rather than assembling protocol frames
    # directly.
    def write_command(self, data: str, *, command_id_hex2: str | None = None) -> None:
        """Send one framed command to the controller."""
        cid = self.default_command_id_hex2 if command_id_hex2 is None else str(command_id_hex2).strip().upper()
        framed = compose_frame(cid, data, checksum_fn=self.checksum_fn)
        self.log_fn(f"TX -> {framed!r}")
        self.ser.write(framed)

    # Maintain an internal byte buffer so fragmented serial reads still produce
    # clean line-based messages.
    def read_line(self, timeout: float = 2.0, keep_terminator: bool = False) -> str:
        """Read one complete line from serial, preserving partial chunks safely."""
        t0 = time.time()

        while time.time() - t0 < timeout:
            if b"\n" in self.buf:
                # Consume exactly one line from the internal buffer.
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

            # Pull more bytes from serial and append to buffer.
            chunk = self.ser.read(256)
            if chunk:
                self.buf += chunk
            else:
                time.sleep(0.01)

        raise TimeoutError("Timed out waiting for serial data")

    # Wait through unrelated telemetry until an actual B6 reply arrives.
    def get_pid_values(self, timeout: float = 2.0) -> dict:
        """Send GET_PID and wait until a valid B6 reply is parsed."""
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
                # Ignore unrelated lines (for example debug telemetry) while waiting for B6.
                continue

        if last_error is not None:
            raise ValueError(f"Failed to parse GET_PID reply before timeout: {last_error}")
        raise TimeoutError("Timed out waiting for GET_PID reply")

    # Same pattern as GET_PID, but for the packed program-config reply.
    def get_program_values(self, timeout: float = 2.0) -> dict:
        """Send GET_PROGRAM and wait until a valid 41 reply is parsed."""
        cmd_bytes = compose_frame("41", "00", checksum_fn=self.checksum_fn)
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
                return parse_program_reply(reply)
            except Exception as e:
                last_error = e
                continue

        if last_error is not None:
            raise ValueError(f"Failed to parse GET_PROGRAM reply before timeout: {last_error}")
        raise TimeoutError("Timed out waiting for GET_PROGRAM reply")

    # Commands and telemetry can be interleaved on the wire, so ignore anything
    # that is not an ACK while waiting for confirmation.
    def write_command_expect_ok_ack(
        self,
        data: str,
        *,
        command_id_hex2: str | None = None,
        timeout: float = 2.0,
        accepted_codes: tuple[str, ...] = ("00",),
    ) -> str:
        """
        Send command and wait for an accepted ACK code.
        Ignores non-ACK lines (e.g. telemetry/debug) until timeout.
        """
        self.write_command(data, command_id_hex2=command_id_hex2)
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            line = self.read_line(timeout=remaining)
            _, code = parse_ack(line)
            # Ignore non-ACK traffic while we wait for command confirmation.
            if not line.startswith("*"):
                continue
            if code in accepted_codes:
                return line
            raise RuntimeError(f"Command failed with ACK: {line}")

        accepted = ", ".join(f"*{c}" for c in accepted_codes)
        raise TimeoutError(f"Timed out waiting for accepted ACK ({accepted})")

    # Preserve untouched PID fields by default so tuning writes do not
    # accidentally zero unrelated controller parameters.
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
        """Build and send SET_PID, returning the immediate ACK line."""
        # If caller did not pass current values, try to read them first so that
        # untouched fields can be preserved by the composer.
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

    # Program writes follow the same read-modify-write idea as PID writes.
    def set_program_values(
        self,
        *,
        power_w: float,
        frequency_khz: float,
        program_id: int | None = 0,
        pulse_width_us: int | None = 0,
        detect_delay_us: int | None = 0,
        current_values: dict | None = None,
        timeout: float = 2.0,
    ) -> str:
        """Build and send SET_PROGRAM, returning the immediate ACK line."""
        if current_values is None:
            try:
                current_values = self.get_program_values(timeout=timeout)
            except Exception:
                current_values = None

        cmd_bytes = compose_set_program_command(
            power_w=power_w,
            frequency_khz=frequency_khz,
            program_id=program_id,
            pulse_width_us=pulse_width_us,
            detect_delay_us=detect_delay_us,
            current_values=current_values,
            checksum_fn=self.checksum_fn,
        )
        self.log_fn(f"TX -> {cmd_bytes!r}")
        self.ser.write(cmd_bytes)
        ack = self.read_line(timeout=timeout)
        return ack
