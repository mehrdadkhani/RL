redis-server &
sleep 1
xterm -e "python control.py" &
python executor.py $1 &
python trainer.py

pkill redis-server