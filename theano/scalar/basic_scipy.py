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


class J1(UnaryScalarOp):
    """
    Bessel function of the 1'th kind
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


class J0(UnaryScalarOp):
    """
    Bessel function of the 0'th kind
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


class I1(UnaryScalarOp):
    """
    Modified Bessel function of the 1'th kind
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
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            #ifndef _I1FUNCDEFINED
            #define _I1FUNCDEFINED
            DEVICE double _i1(double x){
            double ax,ans, y;
            if ((ax=fabs(x)) < 3.75) {
                y=x/3.75;
                y*=y;
                ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
                +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
            } else {
                y=3.75/ax;
                ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
                -y*0.420059e-2));
                ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
                +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
                ans *= (exp(ax)/sqrt(ax));
            }
            return x < 0.0 ? -ans : ans;}
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


class I0(UnaryScalarOp):
    """
    Modified Bessel function of the 1'th kind
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
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            #ifndef _I0FUNCDEFINED
            #define _I0FUNCDEFINED
            DEVICE double _i0(double x){
            /*
            Taken from  NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING
            (ISBN 0-521-43108-5)
            */
            double ax,ans, y;
            if ((ax=fabs(x)) < 3.75) {
                y=x/3.75;
                y*=y;
                ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+
                y*(0.360768e-1+y*0.45813e-2)))));
            }else {
                y=3.75/ax;
                ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1+
                y*(0.225319e-2+y*(-0.157565e-2+
                y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+
                y*(-0.1647633e-1+y*0.392377e-2))))))));
            }
            return ans;}
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
