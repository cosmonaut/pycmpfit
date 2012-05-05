import inspect
import numpy as np
import sys
cimport numpy as cnp
from libc.stdlib cimport free, malloc

cdef extern from "stdlib.h" nogil:
    void *memcpy(void *dest, void *src, size_t n)

# Initialize numpy C API.
cnp.import_array()

# cmpfit header info
cdef extern from "mpfit.h":
    # TODO: add support
    struct mp_par_struct:
        pass
    # TODO: add support
    struct mp_config_struct:
        pass
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
        args = inspect.getargspec(func).args
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
    cdef mp_config *_c_config
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
                 mp_par = None, 
                 mp_config = None, 
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

        self._mp_par = mp_par
        self._mp_config = mp_config
        self._private_data = private_data
        self.result = None

        # Give the user func the data that it needs
        _uf.setParams(self._user_func, self._m, self._n_par, self._xall, self._private_data)
        
    def user_func(self, user_func):
        self._user_func = user_func
        _uf.setFunc(self._user_func)

    def mpfit(self):
        # Note: Always reset _uf values before running
        # Note: we don't actually pass private data through, we handle
        # that on the cython/python side...
        cdef cnp.npy_intp npar_shape[1]
        cdef cnp.npy_intp nparsq_shape[1]
        cdef cnp.npy_intp nfunc_shape[1]
        npar_shape[0] = self._n_par
        nparsq_shape[0] = self._n_par**2

        # TODO: This needs to be copied for the result so it can be freed?? (we free in __del__...)
        self._c_result.xerror = <double *>malloc(self._n_par*sizeof(double))

        # Note: mp_par and mp_config are still not implemented...
        mpfit_ret_code = mpfit(&user_func_wrapper,
                           <int>self._m,
                           <int>self._n_par,
                           self._c_xall,
                           <mp_par *>0,
                           <mp_config *>0,
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
            self.result.version = self._c_result.version.decode(sys.stdout.encoding)
        else:
            raise Exception("C backend has returned error: " + _mpfit_errors[mpfit_ret_code])

        # Return the fitted parameters
        self.result_pars = cnp.PyArray_SimpleNewFromData(1, npar_shape, cnp.NPY_DOUBLE, <void *>self._c_xall)
        
    def config(self, mp_config):
        self._mp_config = mp_config
        
    def par(self, mp_par):
        self._mp_par = mp_par

    def __del__(self):
        free(self._c_xall)
        free(self._c_result.xerror)


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

