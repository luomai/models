gnuplot -c gns.gp gns.txt gns.pdf
./smooth-gns.py > s1.txt
gnuplot -c gns.gp s1.txt s1.pdf
