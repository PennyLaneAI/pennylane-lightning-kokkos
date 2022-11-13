import matplotlib.pyplot as plt

filename = "timings.dat"

with open(filename, "r") as fid:
    tdat = fid.read()

tdat = tdat.splitlines()
tdat.pop(0)

nproc = []
t_adj = []
t_ps = []
for l in tdat:
    n, t0, t1 = l.split()
    nproc.append(int(n))
    t_adj.append(float(t0))
    t_ps.append(float(t1))

fig = plt.figure()

plt.plot(nproc, t_adj, ".-", label="time-adjoint")
plt.plot(nproc, t_ps, ".-", label="time-parameter-shift")

plt.legend()

plt.xlabel("Number of threads")
plt.ylabel("Time (s)")
# plt.yscale("log")
plt.title("Scaling with # of proc")

plt.savefig("scaling.png")
