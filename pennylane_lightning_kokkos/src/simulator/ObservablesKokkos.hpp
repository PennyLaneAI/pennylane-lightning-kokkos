#pragma once

#include <vector>

#include "Error.hpp"
#include "LinearAlgebraKokkos.hpp"
#include "StateVectorKokkos.hpp"

namespace Pennylane::Lightning_Kokkos::Simulators {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */

template <typename T>
class ObservableKokkos
    : public std::enable_shared_from_this<ObservableKokkos<T>> {
  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableKokkos<T> &other) const = 0;

  protected:
    ObservableKokkos() = default;
    ObservableKokkos(const ObservableKokkos &) = default;
    ObservableKokkos(ObservableKokkos &&) noexcept = default;
    ObservableKokkos &operator=(const ObservableKokkos &) = default;
    ObservableKokkos &operator=(ObservableKokkos &&) noexcept = default;

  public:
    virtual ~ObservableKokkos() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorKokkos<T> &sv) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] bool operator==(const ObservableKokkos<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const ObservableKokkos<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam T Floating point type
 */
template <typename T> class NamedObsKokkos final : public ObservableKokkos<T> {
  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast = static_cast<const NamedObsKokkos<T> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsKokkos object, representing a given
     * observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsKokkos(std::string obs_name, std::vector<size_t> wires,
                   std::vector<T> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Lightning_Kokkos::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    void applyInPlace(StateVectorKokkos<T> &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Class models arbitrary Hermitian observables.
 */
template <typename T>
class HermitianObsKokkos final : public ObservableKokkos<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    std::vector<std::complex<T>> matrix_;
    std::vector<size_t> wires_;
    inline static const Lightning_Kokkos::Util::MatrixHasher mh;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast =
            static_cast<const HermitianObsKokkos<T> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsKokkos(MatrixT matrix, std::vector<size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {}

    [[nodiscard]] auto getMatrix() const -> const std::vector<std::complex<T>> {
        return matrix_;
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        // To avoid collisions on cached Kokkos data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(matrix_);
        return obs_stream.str();
    }
    /**
     * @brief Update the statevector sv.
     * @param sv The state vector to update
     */
    void applyInPlace(StateVectorKokkos<T> &sv) const override {
        const auto *matrix_ptr =
            reinterpret_cast<const Kokkos::complex<T> *>(matrix_.data());
        std::vector<Kokkos::complex<T>> conv_matrix =
            std::vector<Kokkos::complex<T>>{matrix_ptr,
                                            matrix_ptr + matrix_.size()};
        sv.applyOperation_std(getObsName(), wires_, false, {}, conv_matrix);
    }
};

/**
 * @brief Class models Tensor product observables
 */
template <typename T>
class TensorProdObsKokkos final : public ObservableKokkos<T> {
  private:
    std::vector<std::shared_ptr<ObservableKokkos<T>>> obs_;
    std::vector<size_t> all_wires_;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast =
            static_cast<const TensorProdObsKokkos<T> &>(other);

        if (obs_.size() != other_cast.obs_.size()) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObsKokkos(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                if (wires.find(wire) != wires.end()) {
                    PL_ABORT("All wires in observables must be disjoint.");
                }
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires_.begin(), all_wires_.end(), std::less{});
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<ObservableKokkos<T>>> obs)
        -> std::shared_ptr<TensorProdObsKokkos<T>> {
        return std::shared_ptr<TensorProdObsKokkos<T>>{
            new TensorProdObsKokkos(std::move(obs))};
    }

    static auto create(std::vector<std::shared_ptr<ObservableKokkos<T>>> obs)
        -> std::shared_ptr<TensorProdObsKokkos<T>> {
        return std::shared_ptr<TensorProdObsKokkos<T>>{
            new TensorProdObsKokkos(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return all_wires_;
    }
    /**
     * @brief Update the statevector sv.
     * @param sv The state vector to update
     */
    void applyInPlace(StateVectorKokkos<T> &sv) const override {
        for (const auto &ob : obs_) {
            ob->applyInPlace(sv);
        }
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Lightning_Kokkos::Util::operator<<;
        std::ostringstream obs_stream;
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }
};

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 */
template <typename T>
class HamiltonianKokkos final : public ObservableKokkos<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<T> coeffs_;
    std::vector<std::shared_ptr<ObservableKokkos<T>>> obs_;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast =
            static_cast<const HamiltonianKokkos<T> &>(other);

        if (coeffs_ != other_cast.coeffs_) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @tparam T1 Floating point type
     * @tparam T2 std::shared_ptr<ObservableKokkos<T>>> type
     * @param coeffs_arg Arguments to construct coefficients
     * @param obs_arg Arguments to construct observables
     */
    template <typename T1, typename T2>
    HamiltonianKokkos(T1 &&coeffs_arg, T2 &&obs_arg)
        : coeffs_{std::forward<T1>(coeffs_arg)}, obs_{std::forward<T2>(
                                                     obs_arg)} {
        PL_ASSERT(coeffs_.size() == obs_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs_arg Argument to construct coefficients
     * @param obs_arg Argument to construct observable terms
     */
    static auto
    create(std::initializer_list<T> coeffs_arg,
           std::initializer_list<std::shared_ptr<ObservableKokkos<T>>> obs_arg)
        -> std::shared_ptr<HamiltonianKokkos<T>> {
        return std::shared_ptr<HamiltonianKokkos<T>>(new HamiltonianKokkos<T>{
            std::move(coeffs_arg), std::move(obs_arg)});
    }

    /**
     * @brief Updates the statevector sv:->sv'.
     * @param sv The statevector to update
     */
    void applyInPlace(StateVectorKokkos<T> &sv) const override {

        StateVectorKokkos<T> buffer(sv.getNumQubits());
        buffer.initZeros();

        for (size_t term_idx = 0; term_idx < coeffs_.size(); term_idx++) {
            StateVectorKokkos<T> tmp(sv);
            obs_[term_idx]->applyInPlace(tmp);
            Lightning_Kokkos::Util::axpy_Kokkos<T>(
                Kokkos::complex<T>{coeffs_[term_idx], 0.0}, tmp.getData(),
                buffer.getData(), tmp.getLength());
        }
        sv.updateData(buffer);
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        auto all_wires = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires.begin(), all_wires.end(), std::less{});
        return all_wires;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {

        using Pennylane::Lightning_Kokkos::Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_ << ", 'observables' : [";
        const auto term_size = coeffs_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }
};

/**
 * @brief Sparse representation of HamiltonianKokkos<T>
 *
 * @tparam T Floating-point precision.
 */
template <typename T>
class SparseHamiltonianKokkos final : public ObservableKokkos<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<std::complex<T>> data_;
    std::vector<std::size_t> indices_; // colum indices
    std::vector<std::size_t> indptr_;  // row_map
    std::vector<std::size_t> wires_;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast =
            static_cast<const SparseHamiltonianKokkos<T> &>(other);

        if (data_ != other_cast.data_ || indices_ != other_cast.indices_ ||
            indptr_ != other_cast.indptr_) {
            return false;
        }

        return true;
    }

  public:
    /**
     * @brief Create a SparseHamiltonian from data, indices and indptr in CSR
     * format.
     * @tparam T1 Complex floating point type
     * @tparam T2 std::vector<std::size_t> type
     * @tparam T3 std::vector<std::size_t> type
     * @tparam T4 std::vector<std::size_t> type
     * @param data_arg Arguments to construct data
     * @param indices_arg Arguments to construct indices
     * @param indptr_arg Arguments to construct indptr
     * @param wires_arg Arguments to construct wires
     */
    template <typename T1, typename T2, typename T3 = T2,
              typename T4 = std::vector<std::size_t>>
    SparseHamiltonianKokkos(T1 &&data_arg, T2 &&indices_arg, T3 &&indptr_arg,
                            T4 &&wires_arg)
        : data_{std::forward<T1>(data_arg)}, indices_{std::forward<T2>(
                                                 indices_arg)},
          indptr_{std::forward<T3>(indptr_arg)}, wires_{std::forward<T4>(
                                                     wires_arg)} {
        PL_ASSERT(data_.size() == indices_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param data_arg Argument to construct data
     * @param indices_arg Argument to construct indices
     * @param indptr_arg Argument to construct ofsets
     * @param wires_arg Argument to construct wires
     */
    static auto create(std::initializer_list<T> data_arg,
                       std::initializer_list<std::size_t> indices_arg,
                       std::initializer_list<std::size_t> indptr_arg,
                       std::initializer_list<std::size_t> wires_arg)
        -> std::shared_ptr<SparseHamiltonianKokkos<T>> {
        return std::shared_ptr<SparseHamiltonianKokkos<T>>(
            new SparseHamiltonianKokkos<T>{
                std::move(data_arg), std::move(indices_arg),
                std::move(indptr_arg), std::move(wires_arg)});
    }

    /**
     * @brief Updates the statevector SV:->SV', where SV' = a*H*SV, and where H
     * is a sparse Hamiltonian.
     */
    void applyInPlace(StateVectorKokkos<T> &sv) const override {
        PL_ABORT_IF_NOT(wires_.size() == sv.getNumQubits(),
                        "SparseH wire count does not match state-vector size");

        StateVectorKokkos<T> d_sv_prime(sv.getNumQubits());

        Lightning_Kokkos::Util::SparseMV_Kokkos<T>(
            sv.getData(), d_sv_prime.getData(), data_, indices_, indptr_);

        sv.updateData(d_sv_prime);
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Lightning_Kokkos::Util::operator<<;
        std::ostringstream ss;
        ss << "SparseHamiltonian: {\n'data' : ";
        for (const auto &d : data_)
            ss << d;
        ss << ",\n'indices' : ";
        for (const auto &i : indices_)
            ss << i;
        ss << ",\n'indptr' : ";
        for (const auto &o : indptr_)
            ss << o;
        ss << "\n}";
        return ss.str();
    }
    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    };
};

} // namespace Pennylane::Lightning_Kokkos::Simulators
