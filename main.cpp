#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <limits>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

using LD = long double;

/// Using following data structure
/// p_vector = {
///     {x_0, f(x_0)},
///     ...
///     {x_n, f(x_n)},
/// }
using p_vector = std::unordered_map<LD, LD>;
using x_vector = std::vector<LD>;
using matrix = typename boost::numeric::ublas::matrix<long double>;
using vector = typename boost::numeric::ublas::vector<long double>;
constexpr LD EPSILON = 1e-10L;

class MathInterpol final {
public:
    explicit MathInterpol(const p_vector& data) {
        SetData(data);
    };

    void SetData(const p_vector& data) {
        if (data.empty()) {
            std::cerr << "BAD_DATA!" << std::endl;
        }
        data_ = data;
        for (const auto& it : data_) {
            xData_.push_back(it.first);
        }

        std::sort(xData_.begin(), xData_.end());

        if (data_.size() != xData_.size()) {
            std::cerr << "BAD_SIZE!" << std::endl;
        }
    }

    LD MethodNewton(const LD& x) const {
        LD answer = data_.at(xData_.at(0));
        LD factor = 1;
        for (size_t n = 1; n < data_.size(); n++) {
            factor *= (x - xData_.at(n - 1));
            x_vector temp;
            for (size_t i = 0; i <= n; i++) {
                temp.push_back(xData_[i]);
            }
            answer += factor*DividedDifference(temp);
        }
        return answer;
    }

    // a + bx + cx^2 + dx^3
    LD QubicSpline(const LD& x) const {
        const size_t size = data_.size() - 1;
        x_vector a(size + 1), b(size), c(size + 1), d(size);
        for (size_t n = 0; n <= size; n++) {
            a[n] = data_.at(xData_.at(n));
        }
        // a is initialized
        c[0] = c[size] = 0;
        matrix A(size - 1, size - 1);
        vector f(size - 1);
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - 1; j++) {
                LD h_i1 = xData_[i + 2] - xData_[i + 1];
                LD h_i = xData_[i + 1] - xData_[i];

                if (std::abs(i - j) > 1) {
                    A(i, j) = 0.0L;
                }
                else if (i - j == 0) {
                    A(i, j) = 2*(h_i1 + h_i);
                }
                else if (i - j == -1) {
                    A(i, j) = h_i1;
                }
                else if (i - j == 1) {
                    A(i, j) = h_i;
                }
            }
        }
        for (int i = 0; i < size - 1; i++) {
            LD h_i1 = xData_[i + 2] - xData_[i + 1];
            LD h_i = xData_[i + 1] - xData_[i];
            f(i) = 3*((a[i + 2] - a[i + 1])/h_i1 - (a[i + 1] - a[i])/h_i);
        }
        vector answ = LU_decomposition(A, f);
        for (int i = 0; i < size - 1; i++) {
            c[i + 1] = answ(i);
        }
        // c is initialized

        for (int i = 0; i < size; i++) {
            LD h_i = xData_[i + 1] - xData_[i];
            d[i] = (c[i + 1] - c[i])/(3*h_i);
            b[i] = (a[i + 1] - a[i])/h_i + h_i*(2*c[i + 1] + c[i])/3.0L;
        }

        if (xData_[0] <= x && x <= xData_[size]) {
            size_t n = 0;
            for (int i = 0; i < size; i++) {
                if (xData_[i] < x && x <= xData_[i + 1]) {
                    n = i;
                }
            }
            return a[n + 1] + b[n]*(x - xData_[n + 1]) + c[n + 1]*(x - xData_[n + 1])*(x - xData_[n + 1]) + d[n]*(x - xData_[n + 1])*(x - xData_[n + 1])*(x - xData_[n + 1]);
        }
        else if (x > xData_[size]) {
            return a[size] + b[size - 1]*(x - xData_[size]);
        }
        return 0;
    }

    LD MNK_LINE(const LD& x) const {
        LD a = 0.0L, b = 0.0L;
        LD xy_sum = 0.0L, x_sum = 0.0L, y_sum = 0.0L, x2_sum = 0.0L;
        const LD n = static_cast<LD>(data_.size());

        for (const auto& it : xData_) {
            xy_sum += it*data_.at(it);
            x_sum += it;
            y_sum += data_.at(it);
            x2_sum += it*it;
        }

        a = (n*xy_sum - x_sum*y_sum)/(n*x2_sum - std::pow(x_sum, 2.0L));
        b = (y_sum - a*x_sum)/n;
        return a*x + b;
    }

private:
    vector low_triangular_solution(const matrix& L, const vector& f) const {
        vector solution(L.size2(), 0.0L);
        solution(0) = f(0);

        for (int i = 1; i < f.size(); i++) {
            long double sum = 0.0L;
            for (int k = 0; k <= i - 1; k++) {
                sum += L(i, k)*solution(k);
            }
            solution(i) = f(i) - sum;
        }

        return solution;
    }

    vector up_triangular_solution(const matrix& U, const vector& f) const {
        const int max_it = (U.size1() - 1);
        vector solution(U.size1(), 0.0L);
        solution(max_it) = f(max_it)/U(max_it, max_it);
        for (int i = max_it - 1; i >= 0; i--) {
            long double sum = 0.0L;
            for (int k = max_it; k > i; k--) {
                sum += U(i, k)*solution(k);
            }
            solution(i) = (f(i) - sum)/U(i, i);
        }
        return solution;
    }

    vector LU_decomposition(const matrix& A, const vector& f) const {
        const auto LU_decompose = [](const matrix& A){
            matrix L(A.size1(), A.size2(), 0.0L), U(A.size1(), A.size2(), 0.0L);

            for (int i = 0; i < A.size1(); i++) {
                L(i, i) = 1.0L;
                U(0, i) = A(0, i);
            }
            for (int i = 1; i < A.size1(); i++) {
                for (int j = 0; j < A.size2(); j++) {
                    if (i <= j) {
                        long double sum = 0.0L;
                        for (int k = 0; k <= i; k++) {
                            sum += L(i, k)*U(k, j);
                        }
                        U(i, j) = A(i, j) - sum;
                    }
                    else {
                        long double sum = 0.0L;
                        for (int k = 0; k <= j; k++) {
                            sum += L(i, k)*U(k, j);
                        }
                        L(i, j) = (A(i, j) - sum)/U(j, j);
                    }
                }
            }
            return std::make_tuple(L, U);
        };

        std::tuple LU = LU_decompose(A);
        const matrix& L = std::get<0>(LU);
        const matrix& U = std::get<1>(LU);

        return up_triangular_solution(U, low_triangular_solution(L, f));
    }

    LD DividedDifference(const x_vector& x_s) const {
        const x_vector::size_type size = x_s.size();
        if (size == 1) {
            return data_.at(x_s.at(0));
        }
        else if (size == 2) {
            return (data_.at(x_s.at(1)) - data_.at(x_s.at(0)))/(x_s.at(1) - x_s.at(0));
        }
        x_vector::size_type last_index = x_s.size() - 1;
        x_vector first(x_s), second(x_s);
        first.erase(first.begin());
        second.pop_back();
        return (DividedDifference(first) - DividedDifference(second))/(x_s.at(last_index) - x_s.at(0));
    }

    p_vector data_;
    x_vector xData_;
};

int main() {

    p_vector data = {
        {1910.0L, 92228496.0L},
        {1920.0L, 106021537.0L},
        {1930.0L, 123202624.0L},
        {1940.0L, 132164569.0L},
        {1950.0L, 151325798.0L},
        {1960.0L, 179323175.0L},
        {1970.0L, 203211926.0L},
        {1980.0L, 226545805.0L},
        {1990.0L, 248709873.0L},
        {2000.0L, 281421906.0L},
    };

    MathInterpol mathInterpol(data);
    mathInterpol.MethodNewton(1915);
    mathInterpol.QubicSpline(1915);
    mathInterpol.MNK_LINE(1915);

    return 0;
}
