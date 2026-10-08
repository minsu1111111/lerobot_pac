"""붓는 동안 음성 정지 ("정지"/"멈춰"/"그만") — 별도 프로세스에서 듣는다.

제어 루프(30fps)와 같은 프로세스에서 Whisper 를 돌리면 변환하는 동안 루프가 끊기므로 프로세스를 나눈다.
프로그램 시작 때 한 번 띄워 Whisper 를 올려 두고(start), 붓기마다 arm() → 루프에서 poll() → disarm().
마이크는 arm 동안만 연다 (IDLE 의 명령 인식과 마이크를 동시에 쓰지 않음).

정지어만 본다: 받아쓴 문장에 정지어가 있으면 stop (voice_command.match_command 와 같은 사전, 정지 우선).
잘못 알아들으면 멈추는 쪽이라 안전하다.

듣는 방식 = 겹치는 창: 최근 window_s(2.0s) 오디오를 hop_s(0.5s)마다 보고, 그중 최근 1s 에 말소리(VAD)가
있으면 Whisper 로 받아쓴다. '말이 끝날 때까지 녹음' 방식은 팬·서보 소음이 계속되면 말 끝을 못 찾아
최대 녹음 길이(4s)를 다 채운 뒤에야 반응했다 (follower1 실측: 정지어 재생 후 4.3s). 겹치는 창이라
정지어가 경계에서 잘려 놓치지 않고, 반응은 소음과 무관하게 ≈ hop + Whisper(1060 fp32 ≈0.3s).
창 만들기는 stt/voice_command.StreamWindows (대기 중 명령 인식 listen_stream 과 공용).
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

STT_DIR = Path(__file__).resolve().parents[1] / "stt"


def _worker(conn, model: str, device: str, mic, window_s: float, hop_s: float, vad_level: int):
    sys.path.insert(0, str(STT_DIR))
    import voice_command as vc

    try:
        # initial_prompt 끔: 잡음을 힌트 문장("물 따라줘. 정지.")으로 받아쓰는 환각이 정지로 이어진다 (시험에서 확인)
        # sample_len 16: 정지어는 짧다. 서보 소음에서 나오는 긴 반복 환각("이곳은 대한민국의 국민들과의…")이
        # 0.68s 씩 잡아먹던 것을 끊는다 (실측).
        cmd = vc.VoiceCommander(model=model, device=device, mic=mic, initial_prompt=None, sample_len=16)
        vad = vc.import_webrtcvad().Vad(vad_level)
        conn.send(("ready", cmd.device))
    except Exception as e:  # 모델·마이크·webrtcvad 문제 → 키보드 정지만 쓰게
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return

    while True:
        msg = conn.recv()  # 대기 (arm 전엔 마이크 닫힘)
        if msg == "quit":
            return
        if msg != "arm":
            continue
        try:
            with vc.StreamWindows(cmd.mic, vad, cmd.frame_ms, window_s, hop_s) as sw:
                while not conn.poll(0.02):  # disarm/quit 가 올 때까지
                    pcm = sw.poll()
                    if pcm is None:
                        continue
                    r = cmd.transcribe_pcm(pcm)
                    if r.text:
                        conn.send(("heard", r.text, r.command, r.stt_s, time.time()))
                        if r.command is not None:  # 같은 소리를 다음 창에서 또 보내지 않게
                            sw.clear()
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))


class VoiceStop:
    def __init__(self, model="small", device="auto", mic=None, window_s=2.0, hop_s=0.5, vad_level=3):
        ctx = mp.get_context("spawn")  # 부모가 CUDA 를 이미 썼으므로 fork 금지
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_worker, args=(child, model, device, mic, window_s, hop_s, vad_level), daemon=True
        )
        self.armed = False
        self.ok = False

    def start(self, timeout=60.0) -> bool:
        self.proc.start()
        if not self.conn.poll(timeout):
            print("[음성정지] 준비 시간 초과 → 키보드 정지만 사용")
            return False
        msg = self.conn.recv()
        self.ok = msg[0] == "ready"
        print(
            f"[음성정지] {'준비됨 (' + msg[1] + ')' if self.ok else '사용 불가 → 키보드 정지만: ' + msg[1]}"
        )
        return self.ok

    def arm(self):
        if self.ok and not self.armed:
            self.conn.send("arm")
            self.armed = True

    def disarm(self):
        if self.armed:
            self.conn.send("disarm")
            self.armed = False
        while self.ok and self.conn.poll():  # 남은 결과 버림
            self.conn.recv()

    def poll(self) -> str | None:
        """정지어를 들었으면 받아쓴 문장, 아니면 None (기다리지 않음)."""
        heard = self.poll_command()
        return heard[1] if heard and heard[0] == "stop" else None

    def poll_command(self) -> tuple[str, str] | None:
        """들은 명령 중 첫 번째 (command, 문장), 없으면 None. 붓는 중 정지 뒤 "돌아가"/"계속" 고르기에도 쓴다."""
        heard = None
        while self.armed and self.conn.poll():
            msg = self.conn.recv()
            if msg[0] == "heard":
                _, text, command, stt_s, _ = msg
                print(f"\n[음성] '{text}' → {command} ({stt_s:.2f}s)")
                if command is not None and (
                    heard is None or command == "stop"
                ):  # 정지가 섞여 있으면 정지 우선
                    heard = (command, text)
            elif msg[0] == "error":
                print(f"\n[음성] 오류 → 키보드만: {msg[1]}")
                self.ok = self.armed = False
        return heard

    def close(self):
        try:
            if self.proc.is_alive():
                self.conn.send("quit")
                self.proc.join(timeout=3)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
