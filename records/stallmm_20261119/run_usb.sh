# stallmm_20261119: = resoff3 run_usb.sh with J changed, port 8985, server policy_server_selrule_stallmm_20261119.py, arms Q400S25 / Q400S40 = Q400B with RDP_STALL_MM=0.0025 / 0.004 (residual ON); nothing else changes
# resoff3_20261119: = resoff_cur_20261117 run_usb.sh with J changed and port 8984 (seed expansion of Q400B0, residual OFF)
# resoff_cur_20261117: = b2 run_usb_q400b.sh (Q400B: Q400, k 999, suppress, fallback 2, 3 mm stall, stallr server, rt bridge) with port 8983 and arm Q400B0 = Q400B with --delta_mode off (residual OFF); nothing else changes
# b2_20261117 AMEND1: = evq_20261117 run_usb.sh (arm Q400B, plain residual, stallr server) with J = b2_20261117 and port 8980, for the seed-matched Q400B base on seeds 20260922/23
# evq_20261117: = evrel run_usb.sh, port 8976, arms Q400B / Q400EV (3 mm stall: SR empty for these arms)
# evrel_20261117: = qconf_20261117 run_usb.sh with server policy_server_selrule_stallr_20261117.py, port 8974, arms BLOCKR / EVREL (RDP_STALL_RATIO=0.0547)
# qconf_20261117: = qlow_rule_20261116 run_usb.sh with server policy_server_selrule_qconf_20261117.py and env RDP_QCONF=1 (switch picks by Q outside promotion need a unanimous ensemble), port 8972; arms CONF400 (Q400 BAL_400_d0), CONF200 (BAL_200_d0), CONFFULL (Q1600 VQB), all with the belief rule.
# qlow_rule_20261116: = belief_parts run_usb.sh with server policy_server_selrule_qzero_20261116.py (RDP_QZERO=1 only for NOQR), port 8971, arms Q400R / G9R / E100R / NOQR = belief rule (k 2, suppress, fallback 2, theta 1.0, continuity term, delay 2) with the given 9-member Q.
# usb_keyed_s13l: = s13j run_stage.sh with server policy_server_selrule_decu_mem_20261108.py --mem-pca mem_pca.npz, ckpt usb15_iqlm9 (IQL on [P_generic, MEM]), arm VQ15IM, port 8909.
# usb_keyed_s13j: = s13g run_stage.sh with the IQL ckpt usb15_iqlf9 (multisocket VGPS objective), arm VQ15I, port 8908; server/rule learned_q unchanged.
#!/bin/bash
# usb_keyed_s13g: = s13e run_stage.sh with residual_HF2.pt, arm VQ15F, ckpt usb15_mcqf9, port 8906 (eval + smoke_eval only).
# usb_keyed_s13e: = s13d run_stage.sh with max_steps 600, port 8903, arm VQ15F, ckpt usb15_mcqc9; smoke_collect = 2 envs, full 600 steps, into presmoke/. Previous usb_keyed_s13b: = s13a run_stage.sh with arm VQ15F and ckpt usb15_mcq9 (Monte Carlo Q). collect modes unused (data reused from s13a).
# usb_keyed_s13a: learned Q selector (VGPS recipe, s8) on F15 + hand-frame residual. Server policy_server_selrule_decu_noclip_20261107.py
# (= decu server + --no-clip-sample), F15 (s12z) + ATMH, K16, select-horizon 24 (candidate = 24-step plan; rotation onset median plan
# step 14, s11g), bridge eval_rdp_bridge_usb_keyed_handframe_20261106.py --delta_mode deploy_gf residual_HF.pt (s12u, unchanged).
# usage: run_stage.sh smoke_collect | collect | smoke_eval | eval
set -u
MODE=$1
J=/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/stallmm_20261119; NW=/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/nowr_cl_20261112; E4=/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/evid4_cl_20261111; DD=/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs
RDP=/home/dyros-recruit/mcy_ws/reactive_diffusion_policy; M=/home/dyros-recruit/mcy_ws/manifeel/manifeel; P=$RDP/rdp_venv/bin/python
PORT=8985; NENV=25
OVR='{"task.env.asset_info_filename":"tacsl_asset_info_usb_symvis_20261028.yaml","task.env.desired_subassemblies":["USB_plug_symvis"]}'
st(){ echo "$(date '+%F %T') $*" >> $J/status.txt; }
menv(){ set +u; source ~/anaconda3/etc/profile.d/conda.sh; conda activate manifeel; source $M/../env.sh >/dev/null 2>&1; }
export CUDA_MODULE_LOADING=LAZY
CK=$(awk '{print $2}' $DD/usb_keyed_s12z_wrtko_dose_20261107/ckpts.sha256)
AT=$RDP/data/outputs/2026.11.04/16.21.22_train_vae_manifeel_usbkeyed_normsplit_atmh/checkpoints/latest.ckpt
RES=$NW/residual_conv.pt; TH=$($P -c "import json; c=json.load(open('$NW/calib.json'))['calib']; print(str(c['theta_xy'])+','+str(c['theta_rot']))")
VQM=$(ls $DD/q_mem_20261110/ckpt/usb15_iqlf9_balm_s[0-8].pt | paste -sd,); VQB=$(ls $DD/q_balance_20261110/ckpt/usb15_iqlf9_bal_s[0-8].pt | paste -sd,)
[ $(echo $VQM | tr ',' '\n' | wc -l) = 9 ] && [ $(echo $VQB | tr ',' '\n' | wc -l) = 9 ] || { st ABORT_CKPT; exit 1; }
Q400=$(ls $DD/data_budget_20261110/ckpt/q_BAL_400_d0_s[0-8].pt | paste -sd,); G9=$(ls $J/ckpt/q_G9_400_s[0-8].pt 2>/dev/null | paste -sd,); E100=$(ls $J/ckpt/q_E100_400_s[0-8].pt 2>/dev/null | paste -sd,)
[ $(echo $Q400 | tr ',' '\n' | wc -l) = 9 ] || { st ABORT_CKPT_Q400; exit 1; }; Q200=$(ls $DD/data_budget_20261110/ckpt/q_BAL_200_d0_s[0-8].pt | paste -sd,); [ $(echo $Q200 | tr ',' '\n' | wc -l) = 9 ] || { st ABORT_CKPT_Q200; exit 1; }
for f in $CK $AT $RES; do [ -f $f ] || { st "ABORT_MISSING $f"; exit 1; }; done
run_one(){ # ARM SEED COHORT OUTROOT TAG [extra bridge args]
  local ARM=$1 SEED=$2 COH=$3 O=$4 TAG=$5; shift 5; local RULE=learned_q SRV
  local DM="--delta_mode deploy_gf --delta_ckpt $RES --evid_theta=$TH"; local SM=""; case $ARM in Q400S25) ARM=Q400B; SM=0.0025;; Q400S40) ARM=Q400B; SM=0.004;; esac   # stallmm
  local OUT=$O/$TAG REC=$O/server_records_$TAG.npz EXTRA=""
  [ -f $OUT/eval_log.json ] && { st "SKIP_EXISTS $TAG"; return 0; }
  local TB="--verifier-ckpt $VQB --evid-est $J/corr_est.npz --evid-theta 1.0 --evid-ext"; SRV=policy_server_selrule_stallmm_20261119.py
  local BX="" SX=""; case $ARM in ONE) ;; ONE_D2) BX="--exec_delay 2";; ONE_D2C) BX="--exec_delay 2"; SX="--coh-lambda 0.02 --coh-steps 4 --coh-delay 2";; ONE_D2R) BX="--exec_delay 2 --rate_limit 0.168,0.145";; ONE_X2) BX="--exec_pregen 2";; NORULE|EXT|RULE|BLOCK|PROMOTE|BOTH4|RELS|Q400R|G9R|E100R|NOQR|CONF400|CONF200|CONFFULL|EVREL|BLOCKR|Q400B|Q400EV) BX="--exec_delay 2"; SX="--coh-lambda 0.02 --coh-steps 4 --coh-delay 2";; *) echo BAD_ARM; return 1;; esac
  local QZ=0 VQ=""; case $ARM in Q400B|Q400EV) ;; Q400R|NOQR|CONF400) VQ=$Q400;; CONF200) VQ=$Q200;; CONFFULL) VQ=$VQB;; G9R) VQ=$G9;; E100R) VQ=$E100;; esac; [ $ARM = NOQR ] && QZ=1; local QC=0; case $ARM in CONF*) QC=1;; esac; local SR=""; case $ARM in EVREL|BLOCKR) SR=0.0547;; esac
  if [ -n "$VQ" ]; then EXTRA="--verifier-ckpt $VQ --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = NORULE ]; then EXTRA="--verifier-ckpt $VQB $SX"
  elif [ $ARM = RULE ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = BLOCK ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 999 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = BLOCKR ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 999 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = Q400B ]; then EXTRA="--verifier-ckpt $Q400 --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 999 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = Q400EV ]; then EXTRA="--verifier-ckpt $Q400 --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 999 --evid-perm-k 1 --evid-suppress --evid-fallback 4 $SX"
  elif [ $ARM = EVREL ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 999 --evid-perm-k 2 --evid-suppress --evid-fallback 4 $SX"
  elif [ $ARM = PROMOTE ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 $SX"
  elif [ $ARM = BOTH4 ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-suppress --evid-fallback 4 $SX"
  elif [ $ARM = RELS ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-perm-k 1 --evid-suppress --evid-fallback 4 $SX"
  elif [ $ARM = EXT ]; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-suppress --evid-fallback 2 --evid-ext $SX"
  elif true; then EXTRA="--verifier-ckpt $VQB --evid-est $E4/corr_est.npz --evid-theta 1.0 --evid-k 2 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = SUPH2 ]; then EXTRA="$TB --evid-k 2 --evid-suppress --evid-fallback 2 $SX"
  elif [ $ARM = SUPH3 ]; then EXTRA="$TB --evid-k 3 --evid-suppress --evid-fallback 3"
  else echo BAD_ARM; return 1; fi
  local X0=$(journalctl -k -b --no-pager 2>/dev/null | grep -c Xid)
  local T0=$(date +%s); cd $RDP
  RDP_STALL_MM=$SM RDP_STALL_RATIO=$SR RDP_QCONF=$QC RDP_QZERO=$QZ RDP_DRAW_SINGLE=1 PYTHONPATH=$RDP:${PYTHONPATH:-} $P $SRV --ldp-ckpt $CK --at-load-dir $AT --task manifeel_pih_c8d_ldp_front_aligned \
    --at-config at_manifeel_pih_l16 --port $PORT --select-k 16 --select-horizon 24 --select-rule $RULE --no-clip-sample --record-path $REC --record-plans \
    --extended-obs-mode growing $EXTRA > $O/server_$TAG.log 2>&1 &
  local SP=$!
  for i in $(seq 1 150); do grep -q listening $O/server_$TAG.log && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  grep -q listening $O/server_$TAG.log || { st FAIL_SERVER_$TAG; kill $SP 2>/dev/null; return 1; }
  ( cd $M; menv
    python eval_rdp_bridge_usb_keyed_rt_20261113.py -o $OUT --port $PORT --max_steps 600 \
      --test_start_seed $COH --latent_refresh 12 --policy_seed $SEED --server_record_path $REC \
      --cfg_overrides "$OVR" --socket_flip $DM $BX "$@" > $O/eval_$TAG.log 2>&1 ) || st NOTE_EVAL_NONZERO_$TAG
  for i in $(seq 1 30); do [ -f $REC ] && break; sleep 2; done
  local PZ=-1; for i in $(seq 1 100); do local CS=$(stat -c %s $REC 2>/dev/null || echo 0); [ "$CS" = "$PZ" ] && [ "$CS" != 0 ] && $P -c "import numpy as np,sys; np.load(sys.argv[1]).files" $REC 2>/dev/null && break; PZ=$CS; sleep 6; done
  kill $SP 2>/dev/null; wait $SP 2>/dev/null
  st "EVAL_DONE $TAG wall_s=$(( $(date +%s) - T0 ))"
  local X1=$(journalctl -k -b --no-pager 2>/dev/null | grep -c Xid); [ "$X1" != "$X0" ] && { st "XID_RUN_EXCLUDED $TAG xid $X0->$X1"; rm -rf $OUT $REC; return 2; }
  grep -q Traceback $O/server_$TAG.log && { st "FAIL_SERVER_TRACEBACK $TAG"; return 1; }
  grep -q Traceback $O/eval_$TAG.log && { st "FAIL_BRIDGE_TRACEBACK $TAG"; return 1; }
  grep -iq "missing key\|unexpected key" $O/server_$TAG.log && { st "STRICT_LOAD_KEYS_FOUND $TAG"; return 1; }
  grep -q "\[stallmm\] RDP_STALL_MM=$SM" $O/server_$TAG.log || { st "STALL_MM_NOT_CONFIRMED $TAG"; return 1; }
  grep -q "clip_sample disabled" $O/server_$TAG.log || { st "NOCLIP_NOT_CONFIRMED $TAG"; return 1; }
  [ -f $OUT/eval_log.json ] || { st "FAIL_NO_OUTPUT $TAG"; return 1; }
}
SEEDS=${SEEDS:-20260921}
case $MODE in
  smoke) for A in ${ARMS:?}; do run_one $A 20260921 76975 $J/smoke smoke_$A --n_test 25 --stop_after 200 && echo OK > $J/STATUS_smoke_$A || echo FAIL > $J/STATUS_smoke_$A; done;;
  eval) for SEED in $SEEDS; do for A in ${ARMS:?}; do for COH in 76700 76725 76750 76775 76800 76825 76850 76875; do T=${A}_s${SEED}_c$COH
          run_one $A $SEED $COH $J/eval $T --n_test $NENV; rc=$?
          [ $rc = 2 ] && { st "HOLD_XID $T"; exit 3; }
          [ $rc != 0 ] && { st "RETRY $T"; rm -rf $J/eval/$T; run_one $A $SEED $COH $J/eval $T --n_test $NENV; rc=$?; [ $rc = 2 ] && { st "HOLD_XID $T"; exit 3; }; [ $rc != 0 ] && st "RUN_FAILED $T"; }; done; done; done;;
esac
