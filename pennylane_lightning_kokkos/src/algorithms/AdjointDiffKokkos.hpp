#pragma once
#include "LinearAlgebraKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include <Kokkos_Core.hpp>
#include <variant>

/// @cond DEV
namespace {
namespace Simulators = Pennylane::Lightning_Kokkos::Simulators;
using namespace Simulators;
} // namespace
/// @endcond

namespace Pennylane::Lightning_Kokkos::Algorithms {

template <class T> class OpsData {
  private:
    size_t num_par_ops_;
    size_t num_nonpar_ops_;
    const std::vector<std::string> ops_name_;
    const std::vector<std::vector<T>> ops_params_;
    const std::vector<std::vector<size_t>> ops_wires_;
    const std::vector<bool> ops_inverses_;
    const std::vector<std::vector<std::complex<T>>> ops_matrices_;

  public:
    /**
     * @brief Construct an OpsData object, representing the serialized
     * operations to apply upon the `%StateVector`.
     *
     * @param ops_name Name of each operation to apply.
     * @param ops_params Parameters for a given operation ({} if optional).
     * @param ops_wires Wires upon which to apply operation
     * @param ops_inverses Value to represent whether given operation is
     * adjoint.
     * @param ops_matrices Numerical representation of given matrix if not
     * supported.
     */
    OpsData(std::vector<std::string> ops_name,
            const std::vector<std::vector<T>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses,
            std::vector<std::vector<std::complex<T>>> ops_matrices)
        : ops_name_{std::move(ops_name)}, ops_params_{ops_params},
          ops_wires_{std::move(ops_wires)},
          ops_inverses_{std::move(ops_inverses)}, ops_matrices_{
                                                      std::move(ops_matrices)} {
        num_par_ops_ = 0;
        for (const auto &p : ops_params) {
            if (!p.empty()) {
                num_par_ops_++;
            }
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Construct an OpsData object, representing the serialized
     operations to apply upon the `%StateVector`.
     *
     * @see  OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<T>> &ops_params,
            const std::vector<std::vector<size_t>> &ops_wires,
            const std::vector<bool> &ops_inverses,
            const std::vector<std::vector<std::complex<T>>> &ops_matrices)
     */
    OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<T>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses)
        : ops_name_{ops_name}, ops_params_{ops_params},
          ops_wires_{std::move(ops_wires)}, ops_inverses_{std::move(
                                                ops_inverses)},
          ops_matrices_(ops_name.size()) {
        num_par_ops_ = 0;
        for (const auto &p : ops_params) {
            if (p.size() > 0) {
                num_par_ops_++;
            }
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Get the number of operations to be applied.
     *
     * @return size_t Number of operations.
     */
    [[nodiscard]] auto getSize() const -> size_t { return ops_name_.size(); }

    /**
     * @brief Get the names of the operations to be applied.
     *
     * @return const std::vector<std::string>&
     */
    [[nodiscard]] auto getOpsName() const -> const std::vector<std::string> & {
        return ops_name_;
    }
    /**
     * @brief Get the (optional) parameters for each operation. Given entries
     * are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<T>>&
     */
    [[nodiscard]] auto getOpsParams() const
        -> const std::vector<std::vector<T>> & {
        return ops_params_;
    }
    /**
     * @brief Get the wires for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getOpsWires() const
        -> const std::vector<std::vector<size_t>> & {
        return ops_wires_;
    }
    /**
     * @brief Get the adjoint flag for each operation.
     *
     * @return const std::vector<bool>&
     */
    [[nodiscard]] auto getOpsInverses() const -> const std::vector<bool> & {
        return ops_inverses_;
    }
    /**
     * @brief Get the numerical matrix for a given unsupported operation. Given
     * entries are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<std::complex<T>>>&
     */
    [[nodiscard]] auto getOpsMatrices() const
        -> const std::vector<std::vector<std::complex<T>>> & {
        return ops_matrices_;
    }

    /**
     * @brief Notify if the operation at a given index is parametric.
     *
     * @param index Operation index.
     * @return true Gate is parametric (has parameters).
     * @return false Gate in non-parametric.
     */
    [[nodiscard]] inline auto hasParams(size_t index) const -> bool {
        return !ops_params_[index].empty();
    }

    /**
     * @brief Get the number of parametric operations.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumParOps() const -> size_t { return num_par_ops_; }

    /**
     * @brief Get the number of non-parametric ops.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumNonParOps() const -> size_t {
        return num_nonpar_ops_;
    }

    /**
     * @brief Get total number of parameters.
     */
    [[nodiscard]] auto getTotalNumParams() const -> size_t {
        return std::accumulate(
            ops_params_.begin(), ops_params_.end(), size_t{0U},
            [](size_t acc, auto &params) { return acc + params.size(); });
    }
};

/**
 * @brief Kokkos-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobianKokkos {
  private:
    using CFP_t = Kokkos::complex<T>;
    using scalar_type_t = T;

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(StateVectorKokkos<T> &sv1,
                               StateVectorKokkos<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            Pennylane::Lightning_Kokkos::Util::getImagOfComplexInnerProduct<T>(
                sv1.getData(), sv2.getData());
    }

    /**
     * @brief Utility method to apply all operations from given
     * `%Pennylane::Lightning_Kokkos::Algorithms::OpsData<T>` object to
     * `%StateVectorKokkos<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void applyOperations(
        StateVectorKokkos<T> &state,
        const Pennylane::Lightning_Kokkos::Algorithms::OpsData<T> &operations,
        bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx] ^ adj,
                                 operations.getOpsParams()[op_idx]);
        }
    }

    /**
     * @brief Utility method to apply the adjoint indexed operation from
     * `%Pennylane::Lightning_Kokkos::Algorithms::OpsData<T>` object to
     * `%StateVectorKokkos<T>`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void applyOperationAdj(
        StateVectorKokkos<T> &state,
        const Pennylane::Lightning_Kokkos::Algorithms::OpsData<T> &operations,
        size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%Pennylane::Lightning_Kokkos::Algorithms::ObsDatum<T>` object to
     * `%StateVectorKokkos<T>`
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorKokkos<T> &state,
                                const ObservableKokkos<T> &observable) {
        using namespace Pennylane::Lightning_Kokkos::Util;
        observable.applyInPlace(state);
    }

    /**
     * @brief Application of observables to given
     * statevectors
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void applyObservables(
        std::vector<StateVectorKokkos<T>> &states,
        const StateVectorKokkos<T> &reference_state,
        const std::vector<std::shared_ptr<ObservableKokkos<T>>> &observables) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_observables = observables.size();
            for (size_t h_i = 0; h_i < num_observables; h_i++) {
                try {
                    states[h_i].updateData(reference_state);
                    applyObservable(states[h_i], *observables[h_i]);
                } catch (...) {
                    ex = std::current_exception();
                }
            }
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Application of adjoint operations to
     * statevectors.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void applyOperationsAdj(
        std::vector<StateVectorKokkos<T>> &states,
        const Pennylane::Lightning_Kokkos::Algorithms::OpsData<T> &operations,
        size_t op_idx) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_states = states.size();
            for (size_t obs_idx = 0; obs_idx < num_states; obs_idx++) {
                try {
                    applyOperationAdj(states[obs_idx], operations, op_idx);
                } catch (...) {
                    ex = std::current_exception();
                }
            }
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline auto getJacIndex(size_t obs_index, size_t tp_index, size_t tp_size)
        -> size_t {
        return obs_index * tp_size + tp_index;
    }

    /**
     * @brief Applies the gate generator for a given parameteric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return T Generator scaling coefficient.
     */
    inline auto applyGenerator(StateVectorKokkos<T> &sv,
                               const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> T {
        return sv.applyGenerator(op_name, wires, adj);
    }

  public:
    AdjointJacobianKokkos() = default;

    /**
     * @brief Utility to create a given operations object.
     *
     * @param ops_name Name of operations.
     * @param ops_params Parameters for each operation in ops_name.
     * @param ops_wires Wires for each operation in ops_name.
     * @param ops_inverses Indicate whether to take adjoint of each operation in
     * ops_name.
     * @param ops_matrices Matrix definition of an operation if unsupported.
     * @return const Pennylane::Lightning_Kokkos::Algorithms::OpsData<T>
     */
    auto createOpsData(
        const std::vector<std::string> &ops_name,
        const std::vector<std::vector<T>> &ops_params,
        const std::vector<std::vector<size_t>> &ops_wires,
        const std::vector<bool> &ops_inverses,
        const std::vector<std::vector<std::complex<T>>> &ops_matrices = {{}})
        -> Pennylane::Lightning_Kokkos::Algorithms::OpsData<T> {
        return {ops_name, ops_params, ops_wires, ops_inverses, ops_matrices};
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorKokkos<T>` object, with
     * one per required observable. The `operations` will be applied to the
     * internal statevector copies, with the operation indices participating in
     * the gradient calculations given in `trainableParams`, and the overall
     * number of parameters for the gradient calculation provided within
     * `num_params`. The resulting row-major ordered `jac` matrix representation
     * will be of size `trainableParams.size() * observables.size()`.
     *
     * @param ref_data Pointer to the statevector data.
     * @param length Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableKokkos for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian(
        const StateVectorKokkos<T> &ref_data, std::vector<std::vector<T>> &jac,
        const std::vector<std::shared_ptr<ObservableKokkos<T>>> &obs,
        const Pennylane::Lightning_Kokkos::Algorithms::OpsData<T> &ops,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        const std::vector<std::string> &ops_name = ops.getOpsName();
        const size_t num_observables = obs.size();

        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        auto tp_it = trainableParams.rbegin();
        const auto tp_rend = trainableParams.rend();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorKokkos<T> lambda(ref_data.getNumQubits());
        lambda.DeviceToDevice(ref_data.getData());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorKokkos<T>> H_lambda(
            num_observables, StateVectorKokkos<T>(lambda.getNumQubits()));
        applyObservables(H_lambda, lambda, obs);

        StateVectorKokkos<T> mu(lambda.getNumQubits());

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "BasisState")) {
                continue;
            }
            if (tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda);
            applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    const T scalingFactor =
                        applyGenerator(mu, ops.getOpsName()[op_idx],
                                       ops.getOpsWires()[op_idx],
                                       !ops.getOpsInverses()[op_idx]) *
                        (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        updateJacobian(H_lambda[obs_idx], mu, jac,
                                       scalingFactor, obs_idx,
                                       trainableParamNumber);
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
        }
    }
};

} // namespace Pennylane::Lightning_Kokkos::Algorithms
