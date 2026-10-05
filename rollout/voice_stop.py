"""붓는 동안 음성 정지 ("정지"/"멈춰"/"그만") — 별도 프로세스에서 듣는다.

제어 루프(30fps)와 같은 프로세스에서 Whisper 를 돌리면 변환하는 동안 루프가 끊기므로 프로세스를 나눈다.
프로그램 시작 때 한 번 띄워 Whisper 를 올려 두고(start), 붓기마다 arm() → 루프에서 poll() → disarm().
마이크는 arm 동안만 연다 (IDLE 의 명령 인식과 마이크를 동시에 쓰지 않음).

정지어만 본다: 받아쓴 문장에 정지어가 있으면 stop (voice_command.match_command 와 같은 사전, 정지 우선).
잘못 알아들으면 멈추는 쪽이라 안전하다. 반응: 말 끝 silence_ms(기본 500ms) + Whisper(1060 fp32 ≈0.3s).
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

STT_DIR = Path(__file__).resolve().parents[1] / "stt"


def _worker(conn, model: str, device: str, mic, silence_ms: int, vad_level: int):
    sys.path.insert(0, str(STT_DIR))
    import voice_command as vc

    try:
        # initial_prompt 끔: 잡음을 힌트 문장("물 따라줘. 정지.")으로 받아쓰는 환각이 정지로 이어진다 (시험에서 확인)
        cmd = vc.VoiceCommander(model=model, device=device, mic=mic, initial_prompt=None)
        webrtcvad = vc.import_webrtcvad()
        vad, cont = webrtcvad.Vad(vad_level), webrtcvad.Vad(max(0, vad_level - 1))
        conn.send(("ready", cmd.device))
    except Exception as e:  # 모델·마이크·webrtcvad 문제 → 키보드 정지만 쓰게
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return

    frame_ms = cmd.frame_ms
    while True:
        msg = conn.recv()  # 대기 (arm 전엔 마이크 닫힘)
        if msg == "quit":
            return
        if msg != "arm":
            continue
        frames = cmd.mic.frames(frame_ms)
        try:
            while not conn.poll():  # disarm/quit 가 올 때까지
                pcm = vc.record_utterance(frames, vad, frame_ms, silence_ms, 0.3, 4.0, cont)
                if pcm is None:
                    continue
                r = cmd.transcribe_pcm(pcm)
                conn.send(("heard", r.text, r.command, r.stt_s, time.time()))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        finally:
            frames.close()


class VoiceStop:
    def __init__(self, model="small", device="auto", mic=None, silence_ms=500, vad_level=2):
        ctx = mp.get_context("spawn")  # 부모가 CUDA 를 이미 썼으므로 fork 금지
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_worker, args=(child, model, device, mic, silence_ms, vad_level), daemon=True
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
        heard = None
        while self.armed and self.conn.poll():
            msg = self.conn.recv()
            if msg[0] == "heard":
                _, text, command, stt_s, _ = msg
                print(f"\n[음성정지] '{text}' → {command} ({stt_s:.2f}s)")
                if command == "stop":
                    heard = text
            elif msg[0] == "error":
                print(f"\n[음성정지] 오류 → 키보드 정지만: {msg[1]}")
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
