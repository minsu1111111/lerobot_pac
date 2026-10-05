"""실제 팔 한 대로 완료 감지 + TTS + STT 시험 (정책 없이, 사람이 텔레옵/손으로 붓는다).

팔로워 한 대를 '물 따르는 손'(rollout 의 오른팔)으로 보고 그 wrist_roll 을 PourDetector 에 넣는다.
  대기: "물 따라줘" (또는 p+Enter) → "물을 따르겠습니다"
  붓기: 1초 가만히(기준선) → 텔레옵으로 기울였다 되돌리기 → "목표량까지 따랐습니다" → post-done 뒤 대기로
  붓는 중 Space·s = 정지, q = 종료, Ctrl+C = 안전 종료

리더 팔이 있으면 텔레옵(--leader-port), 없으면 팔로워 토크를 끄고 손으로 직접 움직인다.
감지 임계값은 rollout/thresholds.json (오른팔 시연 기준). 팔이 다르면 붓는 방향이 반대일 수 있어
--sign auto(기본)는 양쪽 방향을 다 보고 먼저 기울어진 쪽을 쓴다. 끝나면 최대 편차를 출력하니
시연과 다르게 기울이면 --tilt-off / --return-off 로 맞춘다.

  python tools/teleop_pour_check.py                                        # follower1 손으로, 음성
  python tools/teleop_pour_check.py --leader-port /dev/leader2 --leader-id leader2
  python tools/teleop_pour_check.py --no-stt                               # 키보드 p+Enter 로 시작
결과: <OUTPUTS>/teleop_pour/<시각>/pour_NN.csv

주의: 텔레옵 모드에서 종료하면 팔로워 토크가 꺼진다 (lerobot 기본). 팔을 받칠 준비를 할 것.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rollout"))
from pour_rollout import FPS, load_thresholds, make_commander, make_tts  # noqa: E402

from keys import RawKeys  # noqa: E402
from paths import OUTPUTS, THRESHOLDS  # noqa: E402
from pour_detector import DONE, POURING, PourDetector, PourDetectorConfig  # noqa: E402

JOINT = "wrist_roll.pos"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follower-port", default="/dev/follower1")
    ap.add_argument(
        "--follower-id", default="follower1", help="캘리브레이션 파일 이름 (so_follower/<id>.json)"
    )
    ap.add_argument("--leader-port", default=None, help="없으면 토크를 끄고 손으로 움직이는 모드")
    ap.add_argument("--leader-id", default="leader1")
    ap.add_argument(
        "--max-step-deg",
        type=float,
        default=10.0,
        help="텔레옵 프레임당 최대 이동(°). 시작 때 리더와 자세가 달라도 천천히",
    )
    ap.add_argument(
        "--sign", choices=["auto", "+1", "-1"], default="auto", help="붓는 방향 (auto = 먼저 기운 쪽)"
    )
    ap.add_argument("--tilt-off", type=float, default=None, help="붓기 진입 편차(°). 기본 thresholds.json")
    ap.add_argument("--return-off", type=float, default=None, help="복귀 판정 편차(°). 기본 thresholds.json")
    ap.add_argument("--timeout-s", type=float, default=None, help="기본 thresholds.json 의 timeout_suggest_s")
    ap.add_argument("--post-done-s", type=float, default=3.0, help="DONE 뒤 텔레옵을 더 유지하는 시간")
    ap.add_argument("--no-stt", action="store_true", help="음성 대신 키보드 p+Enter")
    ap.add_argument("--no-tts", action="store_true", help="소리 없이 자막만")
    ap.add_argument("--stt-model", default="small")
    ap.add_argument("--stt-mode", choices=["vad", "enter"], default="vad")
    ap.add_argument("--mic", default=None, help="입력 장치 번호 또는 이름 일부")
    ap.add_argument("--out", default=str(OUTPUTS / "teleop_pour"))
    return ap.parse_args(argv)


def make_arms(args):
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = SOFollowerRobotConfig(
        port=args.follower_port, id=args.follower_id, max_relative_target=args.max_step_deg
    )
    follower = SOFollower(cfg)
    leader = None
    if args.leader_port:
        from lerobot.teleoperators.so_leader import SOLeader
        from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

        leader = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id))
    return follower, leader


def pour(args, follower, leader, tts, det_cfgs, run_dir, idx):
    """붓기 1회. 반환: (요약 dict, 종료 요청 여부)."""
    dets = {s: PourDetector(c) for s, c in det_cfgs.items()}
    chosen = None  # 먼저 POURING 에 들어간 방향
    timeout_steps = int(args.timeout_s * FPS)
    rows, step, done_step, stop, quit_req = [], 0, None, None, False
    max_dev = dict.fromkeys(dets, 0.0)
    last_print = 0.0
    prev_raw, wrap = None, 0.0  # wrist_roll 은 ±180° 에서 부호가 뒤집힌다 → 이어 붙여 연속 각도로
    print(f"\n[붓기 {idx}] 1초 가만히 (기준선) → 기울였다 되돌리기 | Space/s 정지, q 종료")
    tts.say("start")
    t0 = time.perf_counter()
    with RawKeys() as keys:
        while True:
            loop_t0 = time.perf_counter()
            t = loop_t0 - t0
            k = keys.poll() or ""
            if "q" in k or "Q" in k:
                stop, quit_req = "quit", True
            elif " " in k or "s" in k or "S" in k:
                stop = "manual"
            if stop:
                tts.say("stopped")
                break

            if leader is not None:
                follower.send_action(leader.get_action())
            raw = float(follower.get_observation()[JOINT])
            if prev_raw is not None and abs(raw - prev_raw) > 180:
                wrap -= 360 if raw > prev_raw else -360
            prev_raw, roll = raw, raw + wrap

            for s, d in dets.items():
                ev = d.update(roll)
                if d.baseline is not None:
                    max_dev[s] = max(max_dev[s], d.dev(roll))
                if ev == POURING and chosen is None:
                    chosen = s
                    print(f"\n[감지] 붓기 시작 (방향 {s}, {t:.1f}s)")
            det = dets[chosen or next(iter(dets))]
            if chosen and det.state == DONE and done_step is None:
                done_step = step
                print(f"\n[감지] 완료 ({t:.1f}s) → 목표량까지 따랐습니다")
                tts.say("done")
            if done_step is not None and step - done_step >= args.post_done_s * FPS:
                stop = "done"
                break
            if step >= timeout_steps:
                stop = "timeout"
                print()
                tts.say("timeout")
                break

            rows.append(
                dict(step=step, t=round(t, 4), raw=raw, roll=roll, state=det.state, chosen=chosen or "")
            )
            if t - last_print >= 0.25:
                last_print = t
                base = "-" if det.baseline is None else f"{det.baseline:6.1f}"
                devs = " ".join(f"{s}:{max_dev[s]:5.1f}" for s in dets)
                print(
                    f"\r[{t:5.1f}s] {det.state:7s} roll {roll:7.1f}  기준 {base}  최대편차 {devs}  "
                    f"(진입 {det.cfg.tilt_off:.0f} / 복귀 {det.cfg.return_off:.0f})",
                    end="",
                    flush=True,
                )
            step += 1
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - loop_t0)))

    path = run_dir / f"pour_{idx:02d}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "t", "raw", "roll", "state", "chosen"])
        w.writeheader()
        w.writerows(rows)
    summ = dict(
        stop=stop,
        sign=chosen,
        done_t=None if done_step is None else round(done_step / FPS, 2),
        max_dev={s: round(v, 1) for s, v in max_dev.items()},
    )
    print(f"\n[결과 {idx}] {summ}  → {path}")
    if chosen is None:
        print(
            f"         붓기 진입 못 함: 최대 편차가 진입 기준 {det.cfg.tilt_off:.0f}° 보다 작음 → --tilt-off 로 낮추기"
        )
    return summ, quit_req


def main(argv=None):
    args = parse_args(argv)
    th = load_thresholds(THRESHOLDS)
    if args.timeout_s is None:
        args.timeout_s = float(th["timeout_suggest_s"])
    signs = ["+1", "-1"] if args.sign == "auto" else [args.sign]
    det_cfgs = {
        s: PourDetectorConfig(
            tilt_off=args.tilt_off if args.tilt_off is not None else th["tilt_off"],
            return_off=args.return_off if args.return_off is not None else th["return_off"],
            hold=th["hold"],
            base_frames=th["base_frames"],
            sign=float(s),
        )
        for s in signs
    }
    run_dir = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    follower, leader = make_arms(args)
    tts = make_tts(not args.no_tts)
    commander = make_commander(not args.no_stt, args.stt_model, args.mic)
    if args.stt_mode == "vad" and hasattr(commander, "listen_auto"):
        listen = commander.listen_auto
    else:
        listen = commander.listen

    try:
        print(f"[팔] 팔로워 {args.follower_port} ({args.follower_id}) 연결 ...")
        if leader is not None:
            follower.connect()
            print(f"[팔] 리더 {args.leader_port} ({args.leader_id}) 연결 → 텔레옵")
            leader.connect()
        else:
            # connect() 는 configure 뒤 토크를 켜서 팔이 예전 목표 위치로 움직일 수 있다 → 버스만 열고 토크 끔
            follower.bus.connect()
            follower.bus.disable_torque()
            if not follower.is_calibrated:
                follower.bus.write_calibration(follower.calibration)
            print("[팔] 리더 없음 → 토크 끔, 손으로 움직이기")
        idx, announce = 0, True
        while True:
            if announce:
                tts.say("ready")
                announce = False
            tts.wait(timeout=8.0)  # 안내 음성이 마이크에 들어가지 않도록
            time.sleep(0.3)
            r = listen()
            if r is None:
                continue
            print(f"[명령] '{r.text}' → {r.command} ({r.source})")
            if r.command == "quit":
                break
            if r.command != "pour":
                if r.command is None and (r.text or "").strip():
                    tts.say("retry")
                continue
            idx += 1
            _, quit_req = pour(args, follower, leader, tts, det_cfgs, run_dir, idx)
            announce = True
            if quit_req:
                break
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 종료")
    finally:
        for arm in (leader, follower):
            try:
                if arm is not None and arm.is_connected:
                    arm.disconnect()
            except Exception as e:
                print(f"[팔] disconnect 실패: {e}")
        tts.wait(timeout=6.0)
        tts.close()


if __name__ == "__main__":
    main()
