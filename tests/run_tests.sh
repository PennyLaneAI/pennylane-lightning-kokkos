rm -f timings.dat
echo \#num_proc t_adj t_ps > timings.dat
for i in 1 2 4 8 16 32 64; do
   python test_gradient.py ${i}
done