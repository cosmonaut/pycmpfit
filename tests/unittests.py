import unittest
import math
import numpy as np
import pycmpfit


def print_results(mp_result, pars, act, test_name):
    print("===========================================")
    print("Test Name: \t %s" % test_name)
    print("Status: \t %i" % mp_result.status)
    print("Chi-square: \t %f" % mp_result.bestnorm)
    print("# of params: \t %i" % mp_result.npar)
    print("# of free: \t %i" % mp_result.nfree)
    print("# pegged: \t %i" % mp_result.npegged)
    print("# iterations: \t %i" % mp_result.niter)
    print("# func evals: \t %i\n" % mp_result.nfev)

    for i in range(mp_result.npar):
        print("P[%i]: %f +/- %f \t (Actual: %f)" % (i, 
                                                    pars[i], 
                                                    mp_result.xerror[i], 
                                                    act[i]))
    print("===========================================")
    
    
def linear_userfunc(m, n, x, private_data):
    # private data is a dict...
    devs = np.zeros((m), dtype=np.float64)
    user_dict = {"deviates": None}

    # f = b - m*x
    for i in range(m):
        f = x[0] - x[1]*private_data["x"][i]
        devs[i] = (private_data["y"][i] - f)/private_data["ey"][i]

    user_dict["deviates"] = devs

    return user_dict


class LinearTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
                           -2.8394297E-01,1.3416969E+00,1.3757038E+00,
                           -1.3703436E+00,4.2581975E-02,-1.4970151E-01,
                           8.2065094E-01], dtype = np.float64)

        self.y = np.array([1.9000429E-01,6.5807428E+00,1.4582725E+00,
                           2.7270851E+00,5.5969253E+00,5.6249280E+00,
                           0.787615,3.2599759E+00,2.9771762E+00,
                           4.5936475E+00], dtype = np.float64)
        
        self.ey = np.zeros((10), dtype = np.float64)
        self.ey[:] = 0.07
        
        self.user_d = {"x": self.x, "y": self.y, "ey": self.ey}
        
        self.m = 10
        self.n = 2
        self.pars = np.array([1.0, 1.0], dtype = np.float64)
        self.act = np.array([3.2, -1.78], dtype = np.float64)
        
        self.fit = pycmpfit.Mpfit(linear_userfunc, 
                                  self.m, 
                                  self.pars, 
                                  private_data = self.user_d)

    def test_fit(self):
        self.fit.mpfit()
        self.assertEqual(self.fit.result.status, 1, msg = "Linear fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 2.756285 and 
                        self.fit.result.bestnorm >= 2.756284, msg = "Linear fit chi-square failure")
        print_results(self.fit.result, self.pars, self.act, "Linear Function")


def quadratic_userfunc(m, n, x, private_data):
    devs = np.zeros((m), dtype = np.float64)
    user_dict = {"deviates": None}

    for i in range(m):
        devs[i] = (private_data["y"][i] - x[0] - x[1]*private_data["x"][i] - (x[2]*(private_data["x"][i])**2))/private_data["ey"][i]

    user_dict["deviates"] = devs

    return user_dict


class QuadraticTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
                           -2.8394297E-01,1.3416969E+00,1.3757038E+00,
                           -1.3703436E+00,4.2581975E-02,-1.4970151E-01,
                           8.2065094E-01], dtype = np.float64)
        self.y = np.array([2.3095947E+01,2.6449392E+01,1.0204468E+01,
                           5.40507,1.5787588E+01,1.6520903E+01,
                           1.5971818E+01,4.7668524E+00,4.9337711E+00,
                           8.7348375E+00], dtype = np.float64)
        
        self.ey = np.zeros((10), dtype = np.float64)
        self.ey[:] = 0.2
        
        self.user_d = {"x": self.x, "y": self.y, "ey": self.ey}
        
        self.m = 10
        self.n = 3
        
        self.pars = np.array([1.0, 1.0, 1.0], dtype = np.float64)
        self.act = np.array([4.7, 0.0, 6.2], dtype = np.float64)

        # For fixed parameter test...
        self.mp_par = [pycmpfit.MpPar(), pycmpfit.MpPar(), pycmpfit.MpPar()]
        self.mp_par[1].fixed = 1
        
        self.fit = pycmpfit.Mpfit(quadratic_userfunc,
                                  self.m,
                                  self.pars,
                                  private_data = self.user_d)
        
    def test_fit(self):
        self.fit.mpfit()
        self.assertEqual(self.fit.result.status, 1, msg = "Quadratic fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 5.679323 and 
                       self.fit.result.bestnorm >= 5.679322, msg = "Quadratic fit chi-square failure")
        print_results(self.fit.result, self.pars, self.act, "Quadratic Function")

    def test_fix_fit(self):
        # Test fixing one parameter using py_mp_par
        self.pars = np.array([1.0, 0.0, 1.0], dtype = np.float64)
        self.fit = pycmpfit.Mpfit(quadratic_userfunc,
                                  self.m,
                                  self.pars,
                                  private_data = self.user_d,
                                  py_mp_par = self.mp_par)
        
        self.fit.mpfit()

        self.assertEqual(self.fit.result.status, 1, msg = "Quadratic fix fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 6.983588 and 
                       self.fit.result.bestnorm >= 6.983587, msg = "Quadratic fix fit chi-square failure")
        
        print_results(self.fit.result, self.pars, self.act, "Quadratic Function (with fixed parameter)")
        
def gaussian_userfunc(m, n, x, private_data):
    devs = np.zeros((m), dtype = np.float64)
    user_dict = {"deviates": None}

    sig2 = x[3]*x[3]
    
    for i in range(m):
        xc = private_data["x"][i] - x[2]
        devs[i] = (private_data["y"][i] - x[1]*math.exp((-0.5*xc*xc)/sig2) - x[0])/private_data["ey"][i]

    user_dict["deviates"] = devs

    return user_dict


class GaussTest(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
                           -2.8394297E-01,1.3416969E+00,1.3757038E+00,
                           -1.3703436E+00,4.2581975E-02,-1.4970151E-01,
                           8.2065094E-01], dtype = np.float64)        
        self.y = np.array([-4.4494256E-02,8.7324673E-01,7.4443483E-01,
                           4.7631559E+00,1.7187297E-01,1.1639182E-01,
                           1.5646480E+00,5.2322268E+00,4.2543168E+00,
                           6.2792623E-01], dtype = np.float64)

        self.ey = np.zeros((10), dtype = np.float64)
        self.ey[:] = 0.5
        
        self.user_d = {"x": self.x, "y": self.y, "ey": self.ey}
        
        self.m = 10
        self.n = 4
        
        self.pars = np.array([0.0, 1.0, 1.0, 1.0], dtype = np.float64)
        self.act = np.array([0.0, 4.70, 0.0, 0.5], dtype = np.float64)

        self.mp_par = list(pycmpfit.MpPar() for i in range(4))
        self.mp_par[0].fixed = 1
        self.mp_par[2].fixed = 1
        
        self.fit = pycmpfit.Mpfit(gaussian_userfunc, 
                                  self.m, 
                                  self.pars, 
                                  private_data = self.user_d)

    def test_fit(self):
        self.fit.mpfit()
        self.assertEqual(self.fit.result.status, 1, msg = "Gaussian fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 10.350032 and 
                        self.fit.result.bestnorm >= 10.350031, msg = "Gaussian fit chi-square failure")
        print_results(self.fit.result, self.pars, self.act, "Gaussian Function")

    def test_fix_fit(self):
        self.pars = np.array([0.0, 1.0, 0.0, 0.1])

        self.fit = pycmpfit.Mpfit(gaussian_userfunc,
                                  self.m,
                                  self.pars,
                                  private_data = self.user_d,
                                  py_mp_par = self.mp_par)

        self.fit.mpfit()

        self.assertEqual(self.fit.result.status, 1, msg = "Gaussian fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 15.516135 and 
                        self.fit.result.bestnorm >= 15.516133, msg = "Quadratic fix fit chi-square failure")

        print_results(self.fit.result, self.pars, self.act, "Gaussian Function (with fixed parameters)")

    def test_fix_limit_fit(self):
        self.pars = np.array([0.0, 1.0, 0.0, 0.1])
        self.mp_par[3].limited[0] = 0
        self.mp_par[3].limited[1] = 1
        self.mp_par[3].limits[0] = -0.3
        self.mp_par[3].limits[1] = +0.2
        
        self.fit = pycmpfit.Mpfit(gaussian_userfunc,
                                  self.m,
                                  self.pars,
                                  private_data = self.user_d,
                                  py_mp_par = self.mp_par)

        self.fit.mpfit()

        self.assertEqual(self.fit.result.status, 1, msg = "Gaussian fit status should be 1")
        self.assertTrue(self.fit.result.bestnorm <= 45.180570 and 
                        self.fit.result.bestnorm >= 45.180568, msg = "Quadratic fix fit chi-square failure")

        print_results(self.fit.result, self.pars, self.act, "Gaussian Function (with fixed and limited parameters)")

if __name__ == '__main__':
    unittest.main()
    
