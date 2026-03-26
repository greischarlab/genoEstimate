# ifndef __ESTIMATE_PMR_ODE_H
# define __ESTIMATE_PMR_ODE_H


#include <RcppArmadillo.h>
#include <vector>
#include <string>


// To avoid many warnings from BOOST
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#pragma clang diagnostic warning "-Wlanguage-extension-token"


using namespace Rcpp;


typedef std::vector<double> VecType;
typedef boost::numeric::odeint::runge_kutta_dopri5<VecType> VecStepperType;

/*
 Switch commenting on next two lines to switch between boost and
 armadillo matrix types:
 */
// typedef boost::numeric::ublas::matrix<double> MatType;
typedef arma::mat MatType;
typedef boost::numeric::odeint::runge_kutta_dopri5<MatType> MatStepperType;



// From https://stackoverflow.com/a/41820991
namespace boost { namespace numeric { namespace odeint {

template <>
struct is_resizeable<arma::vec>
{
    typedef boost::true_type type;
    const static bool value = type::value;
};

template <>
struct same_size_impl<arma::vec, arma::vec>
{
    static bool same_size(const arma::vec& x, const arma::vec& y)
    {
        return x.n_elem == y.n_elem;
    }
};

template<>
struct resize_impl<arma::vec, arma::vec>
{
    static void resize(arma::vec &v1, const arma::vec& v2)
    {
        v1.resize(v2.n_elem);
    }
};

template <>
struct is_resizeable<arma::mat>
{
    typedef boost::true_type type;
    const static bool value = type::value;
};

template <>
struct same_size_impl<arma::mat, arma::mat>
{
    static bool same_size(const arma::mat& x, const arma::mat& y)
    {
        return x.n_rows == y.n_rows && x.n_cols == y.n_cols;
    }
};

template<>
struct resize_impl<arma::mat, arma::mat>
{
    static void resize(arma::mat &v1, const arma::mat& v2)
    {
        v1.resize(v2.n_rows, v2.n_cols);
    }
};

} } } // namespace boost::numeric::odeint






/*
 ==============================================================================
 ==============================================================================
 Observer template classes
 ==============================================================================
 ==============================================================================
 */


template< class C >
struct GreedyObserver
{
    std::vector<C> data;
    std::vector<double> time;
    GreedyObserver() : data(), time() {};

    void operator()(const C& x, const double& t) {
        data.push_back(x);
        time.push_back(t);
        return;
    }

    void clear() {
        data.clear();
        time.clear();
    }
};

/*
 Same as above but for simulations where you do not want to save every
 time step.
 */
template< class C >
struct SelectiveObserver
{
    std::vector<C> data;
    std::vector<double> time;
    size_t save_every;

    SelectiveObserver(const size_t& save_every_)
        : data(), time(), save_every(save_every_), iters(save_every_) {};
    // note: setting `iters` to `save_every_` so that the first time step
    // is always included

    void operator()(const C& x, const double& t) {
        if (iters >= save_every) {
            data.push_back(x);
            time.push_back(t);
            iters = 0;
        }
        iters++;
        return;
    }

    void clear() {
        data.clear();
        time.clear();
    }

protected:

    size_t iters;

};








#endif
