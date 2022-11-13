import timeit
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane_lightning_kokkos import InitArguments
import sys

plt.style.use("bmh")

engine = "lightning.kokkos"
n_samples = 5


def get_time(qnode, params):
    globals_dict = {"grad": qml.grad, "circuit": qnode, "params": params}
    return timeit.timeit(
        "grad(circuit)(params)", globals=globals_dict, number=n_samples
    )


def wires_scaling(n_wires, n_layers, num_threads=None):

    rng = np.random.default_rng(seed=42)
    kokkos_args = InitArguments()
    if num_threads is not None:
        kokkos_args.num_threads = num_threads

    t_adjoint = []
    t_ps = []
    t_backprop = []

    def circuit(params, wires):
        qml.StronglyEntanglingLayers(params, wires=range(wires))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    for i_wires in n_wires:

        dev = qml.device(engine, wires=i_wires, kokkos_args=kokkos_args, batch_obs=True)
        dev_python = qml.device("default.qubit", wires=i_wires)

        circuit_adjoint = qml.QNode(
            lambda x: circuit(x, wires=i_wires), dev, diff_method="adjoint"
        )
        circuit_ps = qml.QNode(
            lambda x: circuit(x, wires=i_wires), dev, diff_method="parameter-shift"
        )
        # circuit_backprop = qml.QNode(lambda x: circuit(x, wires=i_wires), dev_python, diff_method="backprop")

        # set up the parameters
        param_shape = qml.StronglyEntanglingLayers.shape(
            n_wires=i_wires, n_layers=n_layers
        )
        params = rng.standard_normal(param_shape, requires_grad=True)

        t_adjoint.append(get_time(circuit_adjoint, params))
        t_ps.append(get_time(circuit_ps, params))
        # t_backprop.append(get_time(circuit_backprop, params))

    return t_adjoint, t_ps, t_backprop


if __name__ == "__main__":

    num_threads = None
    if len(sys.argv) > 0:
        num_threads = int(sys.argv[1])

    wires_list = [3]
    n_layers = 3
    adjoint_wires, ps_wires, backprop_wires = wires_scaling(
        wires_list,
        n_layers,
        num_threads=num_threads,
    )
    with open("timings.dat", "a") as fid:
        fid.write(f"{num_threads} {adjoint_wires[0]} {ps_wires[0]}\n")
    # Generating the graphic
    fig = plt.figure()

    plt.plot(wires_list, adjoint_wires, ".-", label="adjoint")
    plt.plot(wires_list, ps_wires, ".-", label="parameter-shift")
    # plt.plot(wires_list, backprop_wires, '.-', label="backprop")

    plt.legend()

    plt.xlabel("Number of wires")
    plt.xticks(wires_list)
    plt.ylabel("Log Time")
    plt.yscale("log")
    plt.title("Scaling with wires")

    plt.savefig("scaling.png")
