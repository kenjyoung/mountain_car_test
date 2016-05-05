plot "avgret.dat" using 1:2 title "Avg Return", "avgret.dat" using 1:3 title "Avg Steps"
set title "Average return obtained and steps taken by Expected Sarsa after n-th episode"
set xlabel "Episode number (n)"
set ylabel "Average"
set terminal postscript eps enhanced color
set output "avgret.eps"
replot
