import socket
from typing import Optional


class TcpFrameReader:
    def __init__(self, host: str, port: int, timeout_s: float = 1.0):
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None

    def connect(self) -> bool:
        self.close()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._timeout_s)
            sock.connect((self._host, self._port))
            self._sock = sock
            return True
        except OSError:
            self._sock = None
            return False

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    def recv_exact(self, nbytes: int) -> Optional[bytes]:
        if self._sock is None:
            return None
        chunks = []
        remaining = nbytes
        while remaining > 0:
            try:
                chunk = self._sock.recv(remaining)
            except (socket.timeout, TimeoutError):
                return None
            except OSError:
                self.close()
                return None
            if not chunk:
                self.close()
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class TcpServerFrameReader:
    def __init__(self, host: str, port: int, timeout_s: float = 0.2):
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    def start(self) -> bool:
        if self._server is not None:
            return True
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self._host, self._port))
            server.listen(1)
            server.settimeout(self._timeout_s)
            self._server = server
            return True
        except OSError:
            self.close()
            return False

    def accept_if_needed(self) -> bool:
        if self._server is None:
            if not self.start():
                return False
        if self._conn is not None:
            return True
        try:
            conn, _ = self._server.accept()
            conn.settimeout(self._timeout_s)
            self._conn = conn
            return True
        except (socket.timeout, TimeoutError):
            return False
        except OSError:
            self.close_connection()
            return False

    def close_connection(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._conn.close()
        except OSError:
            pass
        self._conn = None

    def close(self) -> None:
        self.close_connection()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None

    def recv_exact(self, nbytes: int) -> Optional[bytes]:
        if self._conn is None:
            return None
        chunks = []
        remaining = nbytes
        while remaining > 0:
            try:
                chunk = self._conn.recv(remaining)
            except (socket.timeout, TimeoutError):
                return None
            except OSError:
                self.close_connection()
                return None
            if not chunk:
                self.close_connection()
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
