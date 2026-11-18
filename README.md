# qsel_core — 고정 생성기 + Q 선택 + 손 좌표계 보정기 + 정지 종료 규칙 (핵심 코드)

체크포인트·데이터는 들어 있지 않습니다. 실로봇에서 Q와 잔차(보정기)를 **다시 학습**하는 것을 전제로, 실행·학습 로직만 모았습니다.
원본은 시뮬레이터(manifeel / Isaac Gym, 키 USB 삽입) 실험 코드이며, 파일 안의 절대 경로(`/home/dyros-recruit/...`)는 새 환경에 맞게 바꿔야 합니다.

## 구조 한눈에

```
매 12 스텝(refresh)                                   매 스텝
┌──────────────────────────────────────┐            ┌───────────────────────────────────────────┐
│ 생성기: LDP(latent diffusion) → 후보 16개 │            │ 보정기(잔차): 생성기 action + δ            │
│        AT 디코더(촉각으로 재디코딩)        │            │  x,y : 보정량 (hand frame, ±0.2)          │
│ Q (IQL 앙상블 9개): 후보 점수              │──선택──▶   │  rz  : (현재) 보정량 / (신규) 오차 예측+P 법칙 │
│ 정지 규칙: 정지 2회 연속 전에는 "유지" 후보만 │            └───────────────────────────────────────────┘
└──────────────────────────────────────┘
정지 = 24 스텝 손 이동 < 3 mm. 손이 다시 움직이면 정지 카운트 리셋.
전환 후보 = 지난 24 스텝 명령과의 정규화 평균 차이 > 1.0 (Otsu θ).
```

| 파트 | 역할 (option 해석) |
|---|---|
| 생성기 (LDP + AT) | option 후보 |
| 보정기 | 모드 시험 (맞는 모드면 국소 오차를 고쳐 진행) |
| 보정기를 켠 상태의 정지 | option 종료 |
| Q | 다음 option 선택 |

## 파일

| 경로 | 내용 |
|---|---|
| `rdp_side/policy_server_selrule_b2_20261117.py` | 정책 서버: LDP 후보 K개 생성, AT 디코딩, Q 선택, 정지 기반 막기 규칙(`--evid-k 999 --evid-suppress --evid-fallback 2`), 응답에 `evid_done` 포함 |
| `rdp_side/verifier_model.py` | Q 네트워크(Verifier) 정의 |
| `rdp_side/latent_steer.py`, `rdp_side/vms_features_20261019.py` | 서버 보조 (latent 조작, 특징) |
| `manifeel_side/delta_residual_ms_rld_20261024.py` | 잔차 MLP(ResMLP), 상수(RHO 0.2, 입력 차원 LD 100 / IMG 2078 / FF 840) |
| `manifeel_side/delta_residual_gatefree_20261106.py` | 잔차 실행 루프 + **특권 교사**(시뮬레이터 전용 라벨) + 기록 |
| `manifeel_side/delta_residual_handframe_20261106.py` | 손 좌표계 변환 (`to_hand`) |
| `manifeel_side/delta_residual_conv_20261111.py` | 현재 기본 보정기 (수렴 헤드 포함) |
| `manifeel_side/delta_residual_linsign_20261117.py`, `..._b2_20261117.py` | B-2: 전환 뒤 rz 부호를 선형 모델로 |
| `manifeel_side/delta_residual_errsim_20261118.py` | **신규**: rz = 1[g>0]·clip(clip(−0.1745·ê, ±0.3) − 생성기 rz, ±0.2), ê = 부호 있는 yaw 오차 예측 |
| `manifeel_side/delta_residual_rz0_20261118.py` | 대조: rz 출력 0 |
| `manifeel_side/eval_rdp_bridge_usb_keyed_rt_*.py` | 시뮬레이터 브리지 (참고용: 실행 지연 2, refresh 12, 잔차 호출 순서) |
| `manifeel_side/obs_corrector.py`, `dagger_expert.py`, `inhand_teacher_20261026.py`, `usb_wrist_wrench_20261027.py` | 잔차/브리지 의존 모듈 |
| `train/q/` | Q 학습: `build_dataset.py`(refresh 단위 데이터), `train_iql_bal.py`(IQL + 에피소드 균형), `pick_runs.py`·`train_q.py`(데이터 예산) |
| `train/residual/train_nowr.py` | 보정기 학습 (가중 MSE, 손목 힘 이력 입력 가림) |
| `train/rz_errhead/train_err.py` | 신규 rz 오차 헤드 학습 (ê + 게이트) |
| `train/b2_linres/fit.py` | B-2 선형 부호 모델 |
| `rdp_patch/` | reactive_diffusion_policy 포크 변경분(`rdp_fork_tracked.patch`, 기준 커밋 `BASE_COMMIT.txt`) + AT/태스크 설정 |

## 시뮬레이터 실행 설정 (참고)

서버: `--select-k 16 --select-horizon 24 --select-rule learned_q --no-clip-sample --evid-theta 1.0 --evid-k 999 --evid-suppress --evid-fallback 2 --coh-lambda 0.02 --coh-steps 4 --coh-delay 2 --extended-obs-mode growing`
브리지: `--latent_refresh 12 --exec_delay 2 --delta_mode deploy_gf --max_steps 600`

## 실로봇으로 옮길 때 바꿔야 하는 것

1. **보정기 라벨**: `delta_residual_gatefree_20261106.py`의 교사는 시뮬레이터 특권 정보(구멍·플러그 자세)를 씁니다. 실로봇에는 없습니다.
   - x, y: 사람 교정(CR-DAgger식) 또는 성공 기반 라벨.
   - rz (오차 헤드): 성공 순간 손 yaw 기준 라벨은 **삽입 허용 오차(±1~3°) 때문에 장면별 기준이 흔들림**(시뮬레이터 확인: 남은 오차 중앙값 1.6°에서 부호 일치 0.55). 구멍이 고정이면 여러 성공 자세의 평균을 기준으로 쓰는 안(파지 지그 필요)을 검토 중.
2. **Q 데이터**: 실로봇 자율 rollout(후보 중 실행된 것 + 성공 보상). 시뮬레이터 결과상 약 400 runs 미만에서는 Q가 오히려 해로움(200 runs: Q 없음보다 −3.5%p) → 데이터가 적을 때는 확신 낮은 전환 억제 필요.
3. **정지 규칙 상수**: 24 스텝 / 3 mm는 시뮬레이터 제어 주기 기준. 실로봇 제어 주기·임피던스에 맞게 재보정.
4. 절대 경로, wandb 프로젝트명, 포트 번호.

`robot_command_sent: false` — 이 코드는 시뮬레이터에서만 검증되었습니다.
