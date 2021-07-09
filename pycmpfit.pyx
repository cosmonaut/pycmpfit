import inspect
import numpy as np
import sys
cimport numpy as cnp
from libc.stdlib cimport free, malloc

cdef extern from "stdlib.h" nogil:
    void *memcpy(void *dest, void *src, size_t n)

cdef extern from "string.h" nogil:
    void *memset(void *BLOCK, int C, size_t SIZE)

# Initialize numpy C API.
cnp.import_array()

# cmpfit header info
cdef extern from "mpfit.h":
    ctypedef void (*mp_iterproc)()

    struct mp_par_struct:
        int fixed
        int limited[2]
        double limits[2]
        char *parname
        double step
        double relstep
        int side
        int deriv_debug
        double deriv_reltol
        double deriv_abstol

    # NOTE: the user may set the value explicitly; OR, if the passed
    # value is zero, then the "Default" value will be substituted by
    # mpfit().
    struct mp_config_struct:
        double ftol
        double xtol
        double gtol
        double epsfcn
        double stepfactor
        double covtol
        int maxiter
        int maxfev
        int nprint
        int douserscale
        int nofinitecheck
        mp_iterproc iterproc # placeholder...   
        
    struct mp_result_struct:
        double bestnorm
        double orignorm
        int niter
        int nfev
        int status
        int npar
        int nfree
        int npegged
        int nfunc
        double *resid
        double *xerror
        double *covar
        char version[20]

    ctypedef mp_par_struct mp_par
    ctypedef mp_config_struct mp_config
    ctypedef mp_result_struct mp_result
        
    ctypedef int (*mp_func)(int m,
                            int n,
                            double *x,
                            double *fvec,
                            double **dvec,
                            void *private_data)
    extern int mpfit(mp_func funct,
                     int m,
                     int npar,
                     double *xall,
                     mp_par *pars,
                     mp_config *config,
                     void *private_data,
                     mp_result *result)


# TODO: Add success codes
# Error codes with message that can be returned from mpfit
_mpfit_errors = {0: "General input parameter error",
                 -16: "User function produced non-finite values",
                 -17: "No user function was supplied",
                 -18: "No user data points were supplied",
                 -19: "No free parameters",
                 -20: "Memory allocation error",
                 -21: "Initial values inconsistent w constraints",
                 -22: "Initial constraints inconsistent",
                 -23: "General input parameter error",
                 -24: "Not enough degrees of freedom"}
    
cdef class _UserFunc(object):
    cdef public object _fit_func
    cdef public object _private_data
    cdef public object _m
    cdef public object _n
    cdef public object _x
    #cdef cnp.npy_intp shape[1]
    
    def __cinit__(self, user_fit_func):
        self._fit_func = user_fit_func
        self._private_data = None
        # Other data the user func needs from the user.
        self._m = None
        self._n = None
        self._x = None
        #self.shape[0] = <cnp.npy_intp>0

    def setFunc(self, func):
        # deperecated syntax
        #args = inspect.getargspec(func).args
        args = inspect.getfullargspec(func).args
        # TODO: Check argment types...
        if (len(args)) != 4:
            raise Exception("User function does not have correct number of arguments")
        else:
            self._fit_func = func

    # Run the user func and give back dervis and deviates.
    def runFunc(self):
        user_vecs = self._fit_func(self._m, self._n, self._x, self._private_data)
        return user_vecs

    def checkFunc(self):
        if (self._fit_func != None):
            return True
        else:
            return False

    def setX(self, x):
        if type(x) == np.ndarray:
            self._x = x
        
    def setParams(self, f, m, n, x, priv):
        # Type Check and set
        self.setFunc(f)
        if type(m) == int:
            self._m = m
        if type(n) == int:
            self._n = n
        if type(x) == np.ndarray:
            self._x = x
        self._private_data = priv

# Initialize a global UserFunc class -- a trick to keep track of the
# data between python and C...
_uf = _UserFunc(None)

# python wrapped mp_par struct
cdef class MpPar(object):
    cdef public object fixed
    cdef public object limited
    cdef public object limits
    cdef public object parname
    cdef public object step
    cdef public object relstep
    cdef public object side
    cdef public object deriv_debug
    cdef public object deriv_reltol
    cdef public object deriv_abstol

    def __init__(self):
        self.fixed = 0
        self.limited = np.array([0, 0], dtype = np.int32)
        self.limits = np.array([0.0, 0.0], dtype = np.float64)
        self.parname = ""
        self.step = 0.0
        self.relstep = 0.0
        self.side = 0
        self.deriv_debug = 0
        self.deriv_reltol = 0.0
        self.deriv_abstol = 0.0

        
# Python wrapper for mp_config_struct
cdef class MpConfig(object):
    cdef public object ftol
    cdef public object xtol    
    cdef public object gtol
    cdef public object epsfcn
    cdef public object stepfactor
    cdef public object covtol
    cdef public object maxiter
    cdef public object maxfev
    cdef public object nprint
    cdef public object douserscale
    cdef public object nofinitecheck

    def __init__(self):
        # Set everything to 0, cmpfit will handle defaults
        self.ftol = 0.0
        self.xtol = 0.0
        self.gtol = 0.0
        self.epsfcn = 0.0
        self.stepfactor = 0.0
        self.covtol = 0.0
        self.maxiter = 0
        self.maxfev = 0
        self.nprint = 0
        self.douserscale = 0
        self.nofinitecheck = 0
        

# python wrapped mp_result struct
cdef class MpResult(object):
    cdef public object bestnorm
    cdef public object orignorm
    cdef public object niter
    cdef public object nfev
    cdef public object status
    cdef public object npar
    cdef public object nfree
    cdef public object npegged
    cdef public object nfunc
    cdef public object resid
    cdef public object xerror
    cdef public object covar
    cdef public object version
    
    def __init__(self):
        self.bestnorm = 0.0
        self.orignorm = 0.0
        self.niter = 0
        self.nfev = 0
        self.status = 0
        self.npar = 0
        self.nfree = 0
        self.npegged = 0
        self.nfunc = 0
        self.resid = None
        self.xerror = None
        self.covar = None
        self.version = ""


cdef class Mpfit(object):
    cdef double *_c_xall
    cdef mp_par *_c_pars
    cdef mp_config _c_config
    cdef mp_result _c_result
    cdef object _user_func
    cdef object _m
    cdef object _n_par
    cdef object _xall
    cdef object _mp_par
    cdef object _mp_config
    cdef object _private_data
    cdef public object result
    cdef public object result_pars

    def __cinit__(self, user_func, m, cnp.ndarray xall, 
                 py_mp_par = None, 
                 py_mp_config = None, 
                 private_data = None):
        self._user_func = user_func
        
        if type(m) == int:
            self._m = m
        else:
            raise Exception("Incorrect type for m %s should be int." % type(m))
            
        if type(xall) == np.ndarray:
            if cnp.PyArray_NDIM(xall) == 1:
                self._xall = xall
                self._n_par = xall.size
                # Make the C array for mpfit
                self._c_xall = <double *>malloc(self._n_par*sizeof(double))
                self._c_xall = <double *>xall.data
            else:
                raise Exception("xall has incorrect dimensions")
        else:
            raise Exception("xall has incorrect type %s" % type(xall))

        # TODO: This needs to be copied for the result so it can be freed?? (we free in __del__...)
        self._c_result.xerror = <double *>malloc(self._n_par*sizeof(double))
        self._c_result.covar = <double *>malloc(self._n_par*self._n_par*sizeof(double))

        if (py_mp_par):
            self.set_mp_par(py_mp_par)
        else:
            self._mp_par = None
            self._c_pars = <mp_par *>0

        if (py_mp_config):
            self.set_mp_config(py_mp_config)
        else:
            # Defaults
            conf = MpConfig()
            self.set_mp_config(conf)
            
        self._private_data = private_data
        self.result = None

        # Give the user func the data that it needs
        _uf.setParams(self._user_func, self._m, self._n_par, self._xall, self._private_data)
        
    def user_func(self, user_func):
        self._user_func = user_func
        _uf.setFunc(self._user_func)

    def set_mp_par(self, py_mp_par):
        if type(py_mp_par) == list:
            if (len(py_mp_par) == self._n_par):
                try:
                    # Free if already exists?
                    # TODO: test me
                    if (self._c_pars != NULL):
                        free(self._c_pars)
                    # malloc room for all par structs
                    self._c_pars = <mp_par *>malloc(self._n_par*sizeof(mp_par))
                except:
                    raise Exception("Failed to allocate memory for mp_par structs")
                for i, param in enumerate(py_mp_par):
                    # Initialize the mp_par structs...
                    memset(&self._c_pars[i], 0, sizeof(mp_par))
                    if type(param) == MpPar:
                        self._c_pars[i].fixed = param.fixed
                        self._c_pars[i].limited[0] = param.limited[0]
                        self._c_pars[i].limited[1] = param.limited[1]
                        self._c_pars[i].limits[0] = param.limits[0]
                        self._c_pars[i].limits[1] = param.limits[1]
                        # We need this for python 3 ...
                        encoded_parname = param.parname.encode()
                        self._c_pars[i].parname = encoded_parname
                        self._c_pars[i].step = param.step
                        self._c_pars[i].relstep = param.relstep
                        self._c_pars[i].side = param.side
                        if param.side != 3:
                            self._c_pars[i].deriv_debug = param.deriv_debug
                        else:
                            self._c_pars[i].deriv_debug = 0
                            print("deriv_debug set to 0. deriv_debug cannot be on when side = 3")
                            #raise Warning("deriv_debug set to 0 -- cannot be on when side = 3")
                        self._c_pars[i].deriv_reltol = param.deriv_reltol
                        self._c_pars[i].deriv_abstol = param.deriv_abstol
                    else:
                        raise Exception("mp_par element has incorrect type -- should be MpPar")
            else:
                raise Exception("mp_par has incorrect length %i -- should be %i" % (len(py_mp_par), self._n_par))
        else:
            raise Exception("mp_par has incorrect type: %s -- should be list" % type(py_mp_par))
                
        self._mp_par = py_mp_par

    def set_mp_config(self, py_mp_config):
        if (type(py_mp_config) != MpConfig):
            raise Exception("py_mp_config must be MpConfig type")

        self._c_config.ftol = float(py_mp_config.ftol)
        self._c_config.xtol = float(py_mp_config.xtol)
        self._c_config.gtol = float(py_mp_config.gtol)
        self._c_config.epsfcn = float(py_mp_config.epsfcn)
        self._c_config.stepfactor = float(py_mp_config.stepfactor)
        self._c_config.covtol = float(py_mp_config.covtol)
        self._c_config.maxiter = int(py_mp_config.maxiter)
        self._c_config.maxfev = int(py_mp_config.maxfev)
        self._c_config.nprint = int(py_mp_config.nprint)
        self._c_config.douserscale = int(py_mp_config.douserscale)
        self._c_config.nofinitecheck = int(py_mp_config.nofinitecheck)
        self._c_config.iterproc = <mp_iterproc>0
        
    def mpfit(self):
        # Note: Always reset _uf values before running
        # Note: we don't actually pass private data through, we handle
        # that on the cython/python side...
        cdef cnp.npy_intp npar_shape[1]
        cdef cnp.npy_intp nparsq_shape[1]
        cdef cnp.npy_intp nfunc_shape[1]
        cdef cnp.npy_intp covar_shape[2]
        npar_shape[0] = self._n_par
        nparsq_shape[0] = self._n_par**2
        covar_shape[0] = self._n_par
        covar_shape[1] = self._n_par

        # TODO: This needs to be copied for the result so it can be freed?? (we free in __del__...)
        #self._c_result.xerror = <double *>malloc(self._n_par*sizeof(double))

        # Note: mp_config still not implemented...
        mpfit_ret_code = mpfit(&user_func_wrapper,
                           <int>self._m,
                           <int>self._n_par,
                           self._c_xall,
                           self._c_pars,
                           &self._c_config,
                           <void *>0,
                           &self._c_result)

        nfunc_shape[0] = self._c_result.nfunc

        if (mpfit_ret_code > 0):
            self.result = MpResult()
            self.result.bestnorm = self._c_result.bestnorm
            self.result.orignorm = self._c_result.orignorm
            self.result.niter = self._c_result.niter
            self.result.nfev = self._c_result.nfev
            self.result.status = self._c_result.status
            self.result.npar = self._c_result.npar
            self.result.nfree = self._c_result.nfree
            self.result.npegged = self._c_result.npegged
            self.result.nfunc = self._c_result.nfunc
            # resid will be 0 unless the user asks for this... will be implemented later.
            # self.result.resid = cnp.PyArray_SimpleNewFromData(1, 
            #                                                   nfunc_shape, 
            #                                                   cnp.NPY_DOUBLE, 
            #                                                   <void *>self._c_result.resid)
            self.result.xerror = cnp.PyArray_SimpleNewFromData(1, 
                                                               npar_shape, 
                                                               cnp.NPY_DOUBLE, 
                                                               <void *>self._c_result.xerror)
            self.result.covar = cnp.PyArray_SimpleNewFromData(2, 
                                                               covar_shape, 
                                                               cnp.NPY_DOUBLE, 
                                                               <void *>self._c_result.covar)

            self.result.version = self._c_result.version.decode(sys.stdout.encoding)
        else:
            raise Exception("C backend has returned error: " + _mpfit_errors[mpfit_ret_code])

        # Return the fitted parameters
        self.result_pars = cnp.PyArray_SimpleNewFromData(1, npar_shape, cnp.NPY_DOUBLE, <void *>self._c_xall)

    def config(self, mp_config):
        self._mp_config = mp_config

    def __del__(self):
        free(self._c_xall)
        free(self._c_result.xerror)
        free(self._c_result.covar)
        free(self._c_pars)


# public wrapper user function for C library to call
# python user func is called via the _UserFunc class
cdef public api int user_func_wrapper(int m, int n, double *p, double *deviates,
                                      double **derivs, void* private):
    # Pass everything but private data to user func
    # _uf will pass private data...

    # Update *p for the user func (mpfit changes this) (in)
    # Update deviates and derivs for mpfit (out)
        
    # Prepare p to be sent to the user function...
    cdef cnp.npy_intp shape[1]
    shape[0] = n
    
    _uf.setX(cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, <void *>p))

    # run the python user function.
    user_dict = _uf.runFunc()

    # Take the results and update the C side for mpfit...
    # Send the deviates (and dervis eventually) back if they're good
    if type(user_dict['deviates']) == np.ndarray:
        if user_dict['deviates'].dtype == np.float64:
            if len(user_dict['deviates'].shape) == 1:
                # This works!
                #for i in range(m):
                #    deviates[i] = user_dict['deviates'][i]
                memcpy(deviates, cnp.PyArray_DATA(user_dict['deviates']), m*sizeof(double))
                
                # This doesn't!
                #deviates = <double *>(cnp.PyArray_DATA(user_dict['deviates']))
                #deviates = <double *>(user_dict['deviates'].data)
            else:
                print("user vec shape fail")
        else:
            print("Wrong dtype for user dict deviates")
    else:
        print("user vec type fail")        
    # TODO: implement derivs...
    
    return(0)

