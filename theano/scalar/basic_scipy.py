# definition theano.scalar op that have their python implementation taked from scipy
# as scipy is not always available, we treat them separatly
import numpy

import theano
from theano.scalar.basic import (UnaryScalarOp, BinaryScalarOp,
                                 exp, upgrade_to_float,
                                 float_types)
from theano.scalar.basic import (upgrade_to_float_no_complex,
                                 complex_types, discrete_types,
                                 upcast)

imported_scipy_special = False
try:
    import scipy.special
    import scipy.stats
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass


class Erf(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erf(x)
        else:
            super(Erf, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erf(%(x)s);" % locals()
erf = Erf(upgrade_to_float, name='erf')


class Erfc(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfc(x)
        else:
            super(Erfc, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return - gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name='erfc')


class Erfcx(UnaryScalarOp):
    """
    Implements the scaled complementary error function exp(x**2)*erfc(x) in a
    numerically stable way for large x. This is useful for calculating things
    like log(erfc(x)) = log(erfcx(x)) - x ** 2 without causing underflow.
    Should only be used if x is known to be large and positive, as using
    erfcx(x) for large negative x may instead introduce overflow problems.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, sandbox.cuda.opt.local_gpu_elemwise_[0,1] replaces this op
    with sandbox.cuda.elemwise.ErfcxGPU.

    """
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcx(x)
        else:
            super(Erfcx, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * (-cst + (2. * x) * erfcx(x)),

erfcx = Erfcx(upgrade_to_float_no_complex, name='erfcx')


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, sandbox.cuda.opt.local_gpu_elemwise_[0,1] replaces this op
    with sandbox.cuda.elemwise.ErfinvGPU.

    (TODO) Find a C implementation of erfinv for CPU.
    """
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfinv(x)
        else:
            super(Erfinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * cst * exp(erfinv(x) ** 2),

    # TODO: erfinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfinv(%(x)s);" % locals()

erfinv = Erfinv(upgrade_to_float_no_complex, name='erfinv')


class Erfcinv(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcinv(x)
        else:
            super(Erfcinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return - gz * cst * exp(erfcinv(x) ** 2),

    # TODO: erfcinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfcinv(%(x)s);" % locals()

erfcinv = Erfcinv(upgrade_to_float_no_complex, name='erfcinv')


class Gamma(UnaryScalarOp):
    @staticmethod
    def st_impl(x):
        return scipy.special.gamma(x)

    def impl(self, x):
        if imported_scipy_special:
            return Gamma.st_impl(x)
        else:
            super(Gamma, self).impl(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * gamma(x) * psi(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in float_types:
            return """%(z)s = tgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
gamma = Gamma(upgrade_to_float, name='gamma')


class GammaLn(UnaryScalarOp):
    """
    Log gamma function.

    """
    @staticmethod
    def st_impl(x):
        return scipy.special.gammaln(x)

    def impl(self, x):
        if imported_scipy_special:
            return GammaLn.st_impl(x)
        else:
            super(GammaLn, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                lgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
gammaln = GammaLn(upgrade_to_float, name='gammaln')


class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.

    """
    @staticmethod
    def st_impl(x):
        return scipy.special.psi(x)

    def impl(self, x):
        if imported_scipy_special:
            return Psi.st_impl(x)
        else:
            super(Psi, self).impl(x)

    def grad(self, inputs, outputs_gradients):
        raise NotImplementedError()

    def c_support_code(self):
        return (
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(double x){

            /*taken from
            Bernardo, J. M. (1976). Algorithm AS 103:
            Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
            http://www.uv.es/~bernardo/1976AppStatist.pdf */

            double y, R, psi_ = 0;
            double S  = 1.0e-5;
            double C = 8.5;
            double S3 = 8.333333333e-2;
            double S4 = 8.333333333e-3;
            double S5 = 3.968253968e-3;
            double D1 = -0.5772156649;

            y = x;

            if (y <= 0.0)
               return psi_;

            if (y <= S )
                return D1 - 1.0/y;

            while (y < C){
                psi_ = psi_ - 1.0 / y;
                y = y + 1;}

            R = 1.0 / y;
            psi_ = psi_ + log(y) - .5 * R ;
            R= R*R;
            psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

            return psi_;}
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _psi(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
psi = Psi(upgrade_to_float, name='psi')


class Chi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x)) ie. chi2 pvalue (chi2 'survival function').

    C code is provided in the Theano_lgpl repository.
    This make it faster.

    https://github.com/Theano/Theano_lgpl.git

    """

    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, x, k):
        if imported_scipy_special:
            return Chi2SF.st_impl(x, k)
        else:
            super(Chi2SF, self).impl(x, k)
chi2sf = Chi2SF(upgrade_to_float, name='chi2sf')


class J0(UnaryScalarOp):
    """
    Bessel function of the first kind - order 0
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.j0(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(J0, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * -1 * j1(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                j0(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
j0 = J0(upgrade_to_float, name='j0')


class J1(UnaryScalarOp):
    """
    Bessel function of the first kind - order 1
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.j1(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(J1, self).impl(x)

    def grad(self, inp, grads):
        raise NotImplementedError()

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                j1(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
j1 = J1(upgrade_to_float, name='j1')


cephes_chbevl_support_code = """
/*
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1985, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#ifndef _CHBEVLFUNCDEFINED
#define _CHBEVLFUNCDEFINED

double chbevl(double x, double array[], int n)
{
    double b0, b1, b2, *p;
    int i;

    p = array;
    b0 = *p++;
    b1 = 0.0;
    i = n - 1;

    do {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + *p++;
    }
    while (--i);

    return (0.5 * (b0 - b2));
}

#endif
"""

cephes_i0_constants_support_code = """
#ifndef _I0CONSTANTSDEFINED
#define _I0CONSTANTSDEFINED

// For GPU support
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif


/* Chebyshev coefficients for exp(-x) I0(x)
 * in the interval [0,8].
 *
 * lim(x->0){ exp(-x) I0(x) } = 1.
 */
DEVICE double[] _get_I0_A(){
    double _I0_A[] = {
        -4.41534164647933937950E-18,
        3.33079451882223809783E-17,
        -2.43127984654795469359E-16,
        1.71539128555513303061E-15,
        -1.16853328779934516808E-14,
        7.67618549860493561688E-14,
        -4.85644678311192946090E-13,
        2.95505266312963983461E-12,
        -1.72682629144155570723E-11,
        9.67580903537323691224E-11,
        -5.18979560163526290666E-10,
        2.65982372468238665035E-9,
        -1.30002500998624804212E-8,
        6.04699502254191894932E-8,
        -2.67079385394061173391E-7,
        1.11738753912010371815E-6,
        -4.41673835845875056359E-6,
        1.64484480707288970893E-5,
        -5.75419501008210370398E-5,
        1.88502885095841655729E-4,
        -5.76375574538582365885E-4,
        1.63947561694133579842E-3,
        -4.32430999505057594430E-3,
        1.05464603945949983183E-2,
        -2.37374148058994688156E-2,
        4.93052842396707084878E-2,
        -9.49010970480476444210E-2,
        1.71620901522208775349E-1,
        -3.04682672343198398683E-1,
        6.76795274409476084995E-1
    };
    return _I0_A;
}

/* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
 * in the inverted interval [8,infinity].
 *
 * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
 */
DEVICE double[] _get_I0_B(){
    double _I0_B[] = {
        -7.23318048787475395456E-18,
        -4.83050448594418207126E-18,
        4.46562142029675999901E-17,
        3.46122286769746109310E-17,
        -2.82762398051658348494E-16,
        -3.42548561967721913462E-16,
        1.77256013305652638360E-15,
        3.81168066935262242075E-15,
        -9.55484669882830764870E-15,
        -4.15056934728722208663E-14,
        1.54008621752140982691E-14,
        3.85277838274214270114E-13,
        7.18012445138366623367E-13,
        -1.79417853150680611778E-12,
        -1.32158118404477131188E-11,
        -3.14991652796324136454E-11,
        1.18891471078464383424E-11,
        4.94060238822496958910E-10,
        3.39623202570838634515E-9,
        2.26666899049817806459E-8,
        2.04891858946906374183E-7,
        2.89137052083475648297E-6,
        6.88975834691682398426E-5,
        3.36911647825569408990E-3,
        8.04490411014108831608E-1
    };
    return _I0_B;
}
#endif
"""

class I0(UnaryScalarOp):
    """
    Modified Bessel function of the first kind, order 0.
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.i0(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I0, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * i1(x)]

    def c_support_code(self):
        return (
            cephes_chbevl_support_code +
            cephes_i0_constants_support_code +
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            /*
             * Cephes Math Library Release 2.8:  June, 2000
             * Copyright 1984, 1987, 2000 by Stephen L. Moshier
             */

            #ifndef _I0FUNCDEFINED
            #define _I0FUNCDEFINED
            DEVICE double _i0(double x)
            {
                double y;
                _I0_A = _get_I0_A();
                _I0_B = _get_I0_B();

                if (x < 0)
                    x = -x;
                if (x <= 8.0) {
                    y = (x / 2.0) - 2.0;
                    return (exp(x) * chbevl(y, _I0_A, 30));
                }

                return (exp(x) * chbevl(32.0 / x - 2.0, _I0_B, 25) / sqrt(x));
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _i0(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
i0 = I0(upgrade_to_float, name='i0')


class I0e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind, order 0 - exp. scaled
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.i0e(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I0e, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * (i1e(x)-i0e(x))]

    def c_support_code(self):
        return (
            cephes_chbevl_support_code +
            cephes_i0_constants_support_code +
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            /*
             * Cephes Math Library Release 2.8:  June, 2000
             * Copyright 1984, 1987, 2000 by Stephen L. Moshier
             */

            #ifndef _I0EFUNCDEFINED
            #define _I0EFUNCDEFINED
            DEVICE double _i0e(double x)
            {
                double y;
                _I0_A = _get_I0_A();
                _I0_B = _get_I0_B();
                if (x < 0)
                    x = -x;
                if (x <= 8.0) {
                    y = (x / 2.0) - 2.0;
                    return (chbevl(y, _I0_A, 30));
                }

                return (chbevl(32.0 / x - 2.0, _I0_B, 25) / sqrt(x));
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _i0e(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
i0e = I0e(upgrade_to_float, name='i0e')


cephes_i1_constants_support_code = """
#ifndef _I1CONSTANTSDEFINED
#define _I1CONSTANTSDEFINED

/* Chebyshev coefficients for exp(-x) I1(x) / x
 * in the interval [0,8].
 *
 * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
 */
DEVICE double[] _get_I1_A(){
    double _I1_A[] = {
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1
    };
    return _I1_A;
}

/* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
 * in the inverted interval [8,infinity].
 *
 * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
 */
DEVICE double[] _get_I1_A(){
    double _I1_B[] = {
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1
    };
    return _I1_B;
}
#endif
"""

class I1(UnaryScalarOp):
    """
    Modified Bessel function of the first kind, order 1
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.i1(x)

    def grad(self, inp, grads):
        raise NotImplementedError()

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I1, self).impl(x)

    def c_support_code(self):
        return (
            cephes_chbevl_support_code +
            cephes_i1_constants_support_code +
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            /*
             * Cephes Math Library Release 2.8:  June, 2000
             * Copyright 1984, 1987, 2000 by Stephen L. Moshier
             */

            #ifndef _I1FUNCDEFINED
            #define _I1FUNCDEFINED
            double _i1(double x)
            {
                double y, z;
                _I1_A = _get_I1_A();
                _I1_B = _get_I1_B();
                z = fabs(x);
                if (z <= 8.0) {
                    y = (z / 2.0) - 2.0;
                    z = chbevl(y, _I1_A, 29) * z * exp(z);
                }
                else {
                    z = exp(z) * chbevl(32.0 / z - 2.0, _I1_B, 25) / sqrt(z);
                }
                if (x < 0.0)
                    z = -z;
                return (z);
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _i1(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
i1 = I1(upgrade_to_float, name='i1')

class I1e(UnaryScalarOp):
    """
    Modified Bessel function of the first kind, order 1 - exp. scaled
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.i1e(x)

    def grad(self, inp, grads):
        raise NotImplementedError()

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I1e, self).impl(x)

    def c_support_code(self):
        return (
            cephes_chbevl_support_code +
            cephes_i1_constants_support_code +
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            /*
             * Cephes Math Library Release 2.8:  June, 2000
             * Copyright 1984, 1987, 2000 by Stephen L. Moshier
             */

            #ifndef _I1EFUNCDEFINED
            #define _I1EFUNCDEFINED
            double _i1e(double x) {
                double y, z;
                _I1_A = _get_I1_A();
                _I1_B = _get_I1_B();
                z = fabs(x);
                if (z <= 8.0) {
                    y = (z / 2.0) - 2.0;
                    z = chbevl(y, _I1_A, 29) * z;
                }
                else {
                    z = chbevl(32.0 / z - 2.0, _I1_B, 25) / sqrt(z);
                }
                if (x < 0.0)
                    z = -z;
                return (z);
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _i1e(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
i1e = I1e(upgrade_to_float, name='i1e')
