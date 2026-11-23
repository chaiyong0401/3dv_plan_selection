#!/bin/bash
cd /home/dyros-recruit/mcy_ws/real_ms_4070; PY=/home/dyros-recruit/mcy_ws/reactive_diffusion_policy/rdp_venv/bin/python; D=/home/dyros-recruit/mcy_ws/real_ms_4070/data/poolB_20261123; O=/home/dyros-recruit/mcy_ws/real_ms_4070/out/poolB_20261123; mkdir -p $O
export WANDB_MODE=offline
RDP_PHOTO_JITTER_LEGACY_CLIP=1 $PY poolB_20261123.py $D/ckpt_jit/epoch100_backup.ckpt jit $O/jit.json > $O/jit.log 2>&1; echo "jit rc $?" > $O/status
$PY poolB_20261123.py $D/ckpt_tkL/epoch100_backup.ckpt tkL $O/tkL.json > $O/tkL.log 2>&1; echo "tkL rc $?" >> $O/status
$PY /home/dyros-recruit/mcy_ws/qsel_core/records/rr_poolB_20261123/analyze_poolB_20261123.py $O/jit.json $O/tkL.json $O/result.json >> $O/status 2>&1
touch $O/done
