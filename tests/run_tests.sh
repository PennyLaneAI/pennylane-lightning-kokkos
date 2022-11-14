# Avoid warning 
# Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
#   In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
#   For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
#   For unit testing set OMP_PROC_BIND=false
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
export OMP_NUM_THREADS=64
rm -f timings.dat
echo \#num_proc t_adj t_ps > timings.dat
for i in 1 2 4 8 16 32 64; do
   python benchmark_gradient.py ${i}
done
