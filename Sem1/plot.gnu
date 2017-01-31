#! /usr/bin/gnuplot

set terminal png enhanced
set output "plot.png"
set size ratio 0.5
set xlabel "Длина стержня, у.е."
set ylabel "Время, с"
set grid

plot 'outfile.txt' using 1:2 with linespoints
