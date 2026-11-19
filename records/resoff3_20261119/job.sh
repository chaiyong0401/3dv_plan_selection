#!/bin/bash
J=/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/resoff3_20261119
st(){ echo "$(date '+%F %T') $*" >> $J/status.txt; }; P=/home/dyros-recruit/mcy_ws/reactive_diffusion_policy/rdp_venv/bin/python
trap 'touch $J/job.done' EXIT
cd $J; sha256sum -c frozen.sha256 > logs/hash.log 2>&1 || { st ABORT_FROZEN; exit 1; }
X=$(journalctl -k -b 0 --no-pager | grep -c Xid); [ "$X" = 0 ] || { st "FAIL_XID $X"; exit 1; }
FREE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$FREE" -ge 8 ] || { st "FAIL_DISK $FREE"; exit 1; }
st START
SEEDS="20260922 20260923" ARMS=Q400B0 bash run_usb.sh eval; [ $? = 3 ] && { st HELD_XID; exit 3; }
[ $(ls -d eval/Q400B0_s*_c*/eval_log.json 2>/dev/null | wc -l) = 16 ] || { st INCOMPLETE; exit 1; }
grep -q '"delta_mode": "off"' eval/Q400B0_s20260922_c76700/run_metadata.json || { st NOT_OFF; exit 1; }
$P analyze.py > logs/analyze.log 2>&1 && st ANALYZE_DONE || st "FAIL_ANALYZE $(tail -1 logs/analyze.log | cut -c1-300)"
