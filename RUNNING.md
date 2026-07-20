```bash
# train_teacher
nohup python scripts/train_teacher.py > logs/train_teacher.log 2>&1 & echo $! > logs/train_teacher.pid
tail -f logs/train_teacher.log
cat logs/train_teacher.pid
kill -0 $(cat logs/train_teacher.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/train_teacher.pid)

# generate_hazard_labels
python scripts/generate_hazard_labels.py 2>&1 | tee logs/hazard_labels.log

# train_distillation
nohup python scripts/train_distillation.py > logs/distillation.log 2>&1 & echo $! > logs/distillation.pid
tail -f logs/distillation.log
cat logs/distillation.pid
kill -0 $(cat logs/distillation.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/distillation.pid)

# train_baseline
nohup python scripts/train_baseline.py > logs/baseline.log 2>&1 & echo $! > logs/baseline.pid
tail -f logs/baseline.log
cat logs/baseline.pid
kill -0 $(cat logs/baseline.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/baseline.pid)

# train_kd_only.py
nohup python scripts/train_kd_only.py > logs/kd_only.log 2>&1 & echo $! > logs/kd_only.pid
tail -f logs/kd_only.log
cat logs/kd_only.pid
kill -0 $(cat logs/kd_only.pid) && echo "실행 중" || echo "종료됨"
kill $(cat logs/kd_only.pid)

# tmux 마우스 스크롤
vim ~/.tmux.conf
set -g mouse on
tmux source ~/.tmux.conf
```
