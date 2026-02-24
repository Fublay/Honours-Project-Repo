import laser_command_ids as CMD


COMMAND_NAME_BY_ID = {
    CMD.GET_PID: "GET_PID",
    CMD.SET_PID: "SET_PID",
    CMD.PING: "PING",
    CMD.SET_SP: "SET_SP",
    CMD.START: "START",
}


def command_name(command_id_hex2: str) -> str:
    return COMMAND_NAME_BY_ID.get((command_id_hex2 or "").strip().upper(), "UNKNOWN")
