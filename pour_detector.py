"""물 따르기 완료 감지기 (관절 상태 기반, 모델 바깥에서 판단).

물통 팔(오른팔) wrist_roll 한 관절만 본다.
  IDLE    : 시작 base_frames 동안 값을 모아 중앙값 = 기준선
  READY   : 기준선 대비 편차가 tilt_off 를 hold 프레임 연속 넘으면 → POURING
  POURING : 편차가 return_off 안으로 hold 프레임 연속 들어오면 → DONE (1회)
  타임아웃: 전체 step 이 timeout_steps 를 넘으면 상태와 무관하게 DONE (백업)

임계값은 기준선 대비 편차(offset)로 둔다. 방향은 sign(+1/-1)으로 지정한다.
스트리밍(제어 루프)과 오프라인(데이터셋/모델 출력) 양쪽에서 같은 클래스를 쓴다.
"""

from dataclasses import dataclass, field

import numpy as np

IDLE, READY, POURING, DONE = "IDLE", "READY", "POURING", "DONE"


@dataclass
class PourDetectorConfig:
    tilt_off: float  # 기준선 대비 '붓는 중' 진입 편차 (관절 단위)
    return_off: float  # 기준선 대비 '복귀' 판정 편차
    hold: int = 5  # 연속 프레임 수 (30fps 에서 5 = 0.17s)
    base_frames: int = 30  # 기준선 산출 프레임 (30fps 에서 1s)
    sign: float = 1.0  # 붓는 방향 (+1: 값이 커지는 쪽)
    timeout_steps: int | None = None  # None 이면 타임아웃 없음


@dataclass
class PourDetector:
    cfg: PourDetectorConfig
    state: str = IDLE
    step: int = 0
    baseline: float | None = None
    tilt_step: int | None = None  # POURING 진입 step
    done_step: int | None = None  # DONE 진입 step
    done_reason: str | None = None  # "return" | "timeout"
    _buf: list = field(default_factory=list)
    _cnt: int = 0

    def reset(self):
        self.__init__(self.cfg)

    def dev(self, value: float) -> float:
        return self.cfg.sign * (value - self.baseline)

    def update(self, value: float) -> str | None:
        """값 하나 입력. 상태가 바뀐 프레임에만 새 상태 이름을 반환."""
        c = self.cfg
        event = None
        if self.state == IDLE:
            self._buf.append(float(value))
            if len(self._buf) >= c.base_frames:
                self.baseline = float(np.median(self._buf))
                self.state = event = READY
        elif self.state == READY:
            self._cnt = self._cnt + 1 if self.dev(value) > c.tilt_off else 0
            if self._cnt >= c.hold:
                self.state = event = POURING
                self.tilt_step = self.step
                self._cnt = 0
        elif self.state == POURING:
            self._cnt = self._cnt + 1 if self.dev(value) < c.return_off else 0
            if self._cnt >= c.hold:
                self.state = event = DONE
                self.done_step, self.done_reason = self.step, "return"

        if self.state != DONE and c.timeout_steps is not None and self.step >= c.timeout_steps:
            self.state = event = DONE
            self.done_step, self.done_reason = self.step, "timeout"
        self.step += 1
        return event

    def run(self, values) -> "PourDetector":
        """오프라인: 1차원 시퀀스 전체를 흘려 넣는다."""
        for v in values:
            self.update(v)
        return self
