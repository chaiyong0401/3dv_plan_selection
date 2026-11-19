# 새 컴퓨터에서 이어가기 (2026-11-19 기준)

읽는 순서: 이 파일 → [`REVIEW_20261119.md`](REVIEW_20261119.md) (논리 검토, 결정 필요 항목) → [`REAL_ROBOT_MS_PLAN.md`](REAL_ROBOT_MS_PLAN.md) (실로봇 단계) → [`README.md`](README.md) (코드 지도).
Claude Code를 쓰면 [`CLAUDE.md`](CLAUDE.md)가 자동으로 읽힌다.

## 1. 지금 상태

| 항목 | 상태 |
|---|---|
| 시뮬레이터 (키 USB) | 주장 C1–C4 근거 확보. 본 시스템 Q400B 475 / 600, ER3 525 / 600 |
| 실로봇 실행 코드 (`realrobot/`) | CPU 시험 통과. 시뮬레이터 재현은 "근접" (154 대 160, 경계값). `FrankaRobot`은 자리만 있음 |
| 실로봇 실험 | 시작 전. 로봇 명령을 보낸 적 없음 |
| 논문 | 방법 절 초안 v1 (갱신 필요, REVIEW §5) |

## 2. 옮길 것

| 무엇 | 어디 | 크기 | 필요한 경우 |
|---|---|---|---|
| 코드 | 이 저장소 (`git clone`) | 2 MB | 항상 |
| 기록 묶음 `handoff_20261119.tar.gz` | 기존 컴퓨터 `~/mcy_ws/` | 수 MB | 항상. 작업 기억(memory), 계획·관련연구 문서, 허브 페이지, 사전등록·결과 JSON |
| RDP 포크 | 공개 RDP를 `rdp_patch/BASE_COMMIT.txt` 커밋으로 받고 `rdp_patch/rdp_fork_tracked.patch` 적용, `rdp_side/*.py`를 RDP 루트에 복사 | — | 항상 (생성기 학습·서버) |
| 시뮬레이터 체크포인트 | 아래 표 | 3.5 GB | 시뮬레이터 실험을 새 컴퓨터에서 이어갈 때만. **실로봇은 전부 새로 학습하므로 필요 없음** |
| 시뮬레이터 원자료 (`manifeel/data/outputs`) | 기존 컴퓨터 | 수백 GB | 옮기지 않음. 필요한 실험 폴더만 골라서 |

시뮬레이터 체크포인트 (기존 컴퓨터 경로):

| 역할 | 경로 | 크기 |
|---|---|---|
| 생성기 LDP | `reactive_diffusion_policy/data/outputs/2026.11.07/02.37.22_..._s12z_f15_wrtkodose/checkpoints/latest.ckpt` (+ 같은 폴더 `normalizer.pkl`) | 3.4 GB |
| decoder AT | `reactive_diffusion_policy/data/outputs/2026.11.04/16.21.22_..._normsplit_atmh/checkpoints/latest.ckpt` | 0.3 MB |
| Q 앙상블 9개 | `manifeel/manifeel/data/outputs/data_budget_20261110/ckpt/q_BAL_400_d0_s{0..8}.pt` | 49 MB |
| 잔차 | `manifeel/manifeel/data/outputs/nowr_cl_20261112/residual_conv.pt` | 3.5 MB |
| 오차 헤드 (ER3) | `manifeel/manifeel/data/outputs/errsim2_20261118/errnet2.pt` | 7 MB |
| 규칙 보조 | `manifeel/manifeel/data/outputs/evid4_cl_20261111/corr_est.npz` | 40 KB |

복사 예 (새 컴퓨터에서, `OLD` = 기존 컴퓨터 주소):

```bash
git clone git@github.com:chaiyong0401/3dv_plan_selection.git qsel_core
scp OLD:~/mcy_ws/handoff_20261119.tar.gz . && tar xzf handoff_20261119.tar.gz
# Claude Code 작업 기억을 이어 쓰려면 (프로젝트 폴더 이름은 새 컴퓨터의 작업 경로에 따라 다름):
#   cp -r handoff_20261119/memory/* ~/.claude/projects/<새 프로젝트 폴더>/memory/
```

## 3. 환경

| 용도 | 파이썬 / torch | 비고 |
|---|---|---|
| RDP (생성기 학습, 정책 서버, `realrobot/`) | 3.10 / torch 1.13.1 + cu117 | RDP 저장소의 설치 절차 |
| 시뮬레이터 (manifeel, Isaac Gym) | 3.8 / torch 2.4.1 + cu121 | 실로봇에는 필요 없음 |
| GPU | 기존: RTX 4070 (후보 16개 한 번에 138 ms → 교체 3스텝 뒤) | 새 GPU에서 `realrobot/timing_test.py`로 다시 측정 |

새 컴퓨터에서 고칠 경로:

- `realrobot/rr_server.py`, `timing_test.py`, `policy_core.py`의 `/home/dyros-recruit/...` (RDP 위치, realrobot 위치).
- `manifeel_side/`, `train/` 파일 안의 절대 경로와 wandb 프로젝트 이름.

먼저 돌려 볼 것 (GPU·로봇 없이):

```bash
cd realrobot && python test_realrobot_20261119.py     # ALL OK 가 나와야 함
```

## 4. 다음 순서

1. REVIEW §3의 남은 결정 2건 (D3 3 mm 설명, D4 재현 차이 추적). D1·D2는 11-19에 결정됨.
2. 실로봇 정보 확정 (PLAN §8: 제어기, GPU, 소켓 치수·공차, 성공 감지, 리셋, 원격 조작).
3. PLAN §4 단계 0–1: `FrankaRobot` 작성, 지연 측정, 명령 재생으로 기록 재현.
4. 시연 50 → 1차 생성기 → 넘겨받기 수집 200–400 → 생성기 재학습 + 잔차 → **자율 rollout 400 (Q용)** → Q → 평가 480회.
   로봇 시간 합계 약 40 h (시연 3 + 넘겨받기 5–10 + 자율 10 + 평가 12 + 여유).
5. 평가 전에 문서로 고정: 시작 조건 80개, 정지 상수(로그 척도 Otsu), 판정 표, 제외 규칙.

## 5. 지켜 온 규칙

- 실험 전에 판정 기준을 JSON으로 고정하고 sha256을 남긴다. 결과를 본 뒤 기준을 바꾸지 않는다. 실패도 그대로 기록한다.
- 성공률과 시간(RMST)을 함께 보고한다.
- 선택 계층(Q, 규칙)에는 과제 변수·모드 라벨을 넣지 않는다.
- 시뮬레이터에만 통하는 튜닝은 하지 않는다. 실로봇으로 옮겨지는 것만 한다.
- seed 1개 결과는 seed 간 흔들림(±2–3 / 200)과 비교해 해석한다.
- 로봇에 명령을 보내기 전까지 모든 기록에 `robot_command_sent: false`.
