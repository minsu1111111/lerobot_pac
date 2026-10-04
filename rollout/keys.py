"""제어 루프 중 비차단 키 입력 (추가 의존성 없음: termios + select).

  with RawKeys() as keys:
      k = keys.poll()   # 그동안 눌린 문자들(str) 또는 None (기다리지 않음)

cbreak 모드라 Enter 없이 한 글자씩 읽히고, Ctrl+C(SIGINT) 는 그대로 동작한다.
나갈 때 터미널 설정을 원래대로 되돌리고 남은 입력은 버린다 (STT 의 Enter 대기에 섞이지 않게).
stdin 이 터미널이 아니면(파이프·백그라운드 실행) 조용히 꺼진다 — 이때 정지는 Ctrl+C 만 된다.
"""

import os
import select
import sys


class RawKeys:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self.enabled = False
        self._old = None

    def __enter__(self):
        try:
            import termios
            import tty

            self.fd = self.stream.fileno()
            if os.isatty(self.fd):
                self._old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.enabled = True
        except (ImportError, OSError, ValueError, AttributeError):
            self.enabled = False
        return self

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        r, _, _ = select.select([self.fd], [], [], 0)
        if not r:
            return None
        return os.read(self.fd, 64).decode(errors="ignore") or None

    def __exit__(self, *exc):
        if self._old is not None:
            import termios

            termios.tcflush(self.fd, termios.TCIFLUSH)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)
            self._old = None
        self.enabled = False
