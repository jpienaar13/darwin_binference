from binference.simulators import simulate_interpolated
from binference.template_source import TemplateSource
from binference.utils import read_neyman_threshold


from blueice.likelihood import UnbinnedLogLikelihood, LogAncillaryLikelihood, LogLikelihoodSum
from blueice.inference import bestfit_scipy, one_parameter_interval

from inference_interface import toydata_to_file, toydata_from_file, structured_array_to_dict, dict_to_structured_array


import numpy as np
import scipy.stats as sps
from copy import deepcopy
from itertools import product
from scipy.interpolate import interp1d
from os import path
import pkg_resources

minimize_kwargs = {'method': 'Powell', 'options': {'maxiter': 10000000}}

parameter_uncerts = dict(
    cevns=0.04,
    atnu=0.2,
    solarnu = 0.03,
    er=0.1,
)
science_slice_args = [{'slice_axis': 0, 'sum_axis': False, 'slice_axis_limits': [3.01, 149.9]}]


def get_template_filename(fname=""):
    """
        Function to search for the template in two paths-- 1st the common binference folder
        and then the git path
    """
    #midway_path = "/project2/lgrandi/binference_common/nt_projection_templates/v8/"
    midway_path = "/home/jpienaar/binference/binference/data/"
    if path.exists(midway_path):
        return midway_path + fname
    else:
        return pkg_resources.resource_filename("binference", "data/" + fname)

template_names = dict(
        er = "ER_DarwinTemplate.hdf",
        signal = "WIMP_{wimp_mass:d}GeV_DarwinTemplate.hdf",
        cevns = "CEvNS_Solar_DarwinTemplate.hdf",
        atnu = "CEvNS_ATM_DarwinTemplate.hdf",
        solarnu = "Solar_Nu_DarwinTemplate.hdf",
        )

#Find resources:
for k,i in template_names.items():
    template_names[k] = get_template_filename(i)


def get_wimp_signal_config(wimp_mass=50):
    #set interaction etc. here for off-nominal
    ret = dict(
            fname = template_names["signal"],
            histname="hmc",
            #wimp_mass=wimp_mass, 
            parameter_list=["wimp_mass"],
            named_parameters=["wimp_mass"]
            )
    return ret


#config file defining likelihood:#

def get_likelihood_config(signal_config = get_wimp_signal_config(), livetime_days=1.):
    default_signal_config = dict(
            name="signal",
            label="signal",
            templatename=template_names["signal"],
            histname="hmc",
            )
    default_signal_config.update(signal_config)
    er_pars = []
    signal_pars = signal_config.get("parameter_list",[])
    all_pars = set(er_pars+signal_pars)
    er_ignore_pars = all_pars.difference(set(er_pars))
    signal_ignore_pars = all_pars.difference(set(signal_pars))


    source_configs = [
        dict(
            name="er",
            label="Electronic Recoil",
            templatename=template_names["er"],
            ignore_parameters=list(er_ignore_pars),
            named_parameters = [],
            histname="hmc",
            ),
        dict(
            name="cevns",
            label="8B Solar CEvNS",
            templatename=template_names["cevns"],
            ignore_parameters=list(all_pars),
            named_parameters = [],
            histname="hmc",
            ),
        dict(
            name="atnu",
            label="Atmospheric CEvNS",
            templatename=template_names["atnu"],
            ignore_parameters=list(all_pars),
            named_parameters = [],
            histname="hmc",
            ),
        dict(
            name="solarnu",
            label="Solar Nu",
            templatename=template_names["solarnu"],
            ignore_parameters=list(all_pars),
            named_parameters = [],
            histname="hmc",
            ),
            ]

    source_configs.append(default_signal_config)


    ll_config = dict(
            analysis_space=[("cs1",np.linspace(3,150,150-3+1)),
                             ("logcs2",np.linspace(2, 4.75,111))], 
            slice_args = science_slice_args,
            sources = source_configs,
            default_source_class=TemplateSource, 
            livetime_days=livetime_days,
            in_events_per_bin=True,
            log10_bins=[],
            )
    for signal_par in signal_pars:
        ll_config[signal_par] = signal_config.get(signal_par,0.)
    return ll_config

class InferenceObject:
    def __init__(self, wimp_mass = 50, livetime = 1.,
            ll_config_overrides={},
            limit_threshold = None,
            confidence_level = 0.9,
            wimp_masses = None,
            toydata_file = None,
            toydata_mode = "write",
            **kwargs):
        
        signal_config = get_wimp_signal_config(wimp_mass=wimp_mass)
        
        if limit_threshold is None: 
            limit_threshold_function = lambda x,dummy:sps.chi2(1).isf(0.1)
        else: 
            print("loading limit_threshold {:s}, confidence level {:.2f}".format(limit_threshold, confidence_level))

            signal_expectations, thresholds, nominal_signal_expectation = read_neyman_threshold(limit_threshold,
                    locals(), confidence_level = confidence_level)

            print("loaded threshold")
            print("signal_expectations",signal_expectations)
            print("thresholds",thresholds)
            print("nominal_signal_expectation", nominal_signal_expectation)
            ltf = interp1d(signal_expectations/nominal_signal_expectation , thresholds,
                    bounds_error = False, fill_value = sps.chi2(1).isf(0.1))
            def limit_threshold_function(x,cl):
                return ltf(x)

        self.limit_threshold_function = limit_threshold_function
        ll_config = get_likelihood_config(signal_config)
        ll_config["wimp_mass"] = wimp_mass

        ll_config.update(ll_config_overrides)
        ll_config["livetime_days"] = livetime
        ll = UnbinnedLogLikelihood(ll_config)

        for parameter in ["er","cevns","atnu","solarnu"]:
            ll.add_rate_parameter(parameter, log_prior=sps.norm(1,parameter_uncerts[parameter]).logpdf)
        ll.add_rate_parameter("signal")

        if wimp_masses is not None: 
            ll.add_shape_parameter("wimp_mass",wimp_masses)

        ll.prepare()

        dtype = [("cs1",float),("logcs2",float)]
        dummy_data = np.zeros(1,dtype)
        dummy_data["cs1"] = 50.
        dummy_data["logcs2"] = 2.
        ll.set_data(dummy_data)

        self.lls = [ll]
        self.ll = LogLikelihoodSum(self.lls)
        self.dataset_names = ["data_sci","ancillary_measurements","generate_args"]

        self.toydata_mode = toydata_mode
        self.toydata_file = toydata_file


        if toydata_mode =="none":
            self.rgs =[simulate_interpolated(ll) for ll in self.lls]
        elif toydata_mode =="write":
            self.rgs =[simulate_interpolated(ll) for ll in self.lls]
            self.datasets_array = []
        elif toydata_mode =="read":
            self.datasets_array, dataset_names = toydata_from_file(toydata_file)
            assert self.dataset_names == dataset_names
            self.toydata_index = 0

    def assign_data(self,datas):
        for data, ll in zip(datas, self.lls):
            ll.set_data(data)

    def simulate_and_assign_data(self, generate_args = {}):
        datas = [rg.simulate(**generate_args) for rg in self.rgs]
        self.assign_data(datas)
        return datas
    
    def simulate_and_assign_measurements(self,generate_args={}):
        ret = dict()
        for parameter_name in ["er","cevns","atnu","solarnu"]:
            parameter_uncert = parameter_uncerts[parameter_name]
            parameter_mean = generate_args.get(parameter_name+"_rate_multiplier",1)
            parameter_meas = max(0,sps.norm(parameter_mean, parameter_uncert).rvs())
            ret[parameter_name+"_rate_multiplier"] = parameter_meas
            self.lls[0].rate_parameters[parameter_name]=sps.norm(parameter_meas,parameter_uncert).logpdf
        return ret
    
    def assign_measurements(self,ancillary_measurements):
        for parameter_name in ["er","cevns","atnu","solarnu"]:
            parameter_uncert = parameter_uncerts[parameter_name]
            parameter_meas = ancillary_measurements[parameter_name+"_rate_multiplier"]
            self.lls[0].rate_parameters[parameter_name]=sps.norm(parameter_meas,parameter_uncert).logpdf

    def llr(self, extra_args={}, extra_args_null={"signal_rate_multiplier":0.},guess={}):
        extra_args_null_total = deepcopy(extra_args)
        extra_args_null_total.update(extra_args_null)
        res1, llval1 = bestfit_scipy(self.ll, guess=guess,minimize_kwargs=minimize_kwargs, **extra_args)
        res0, llval0 = bestfit_scipy(self.ll, guess=guess,minimize_kwargs=minimize_kwargs,**extra_args_null_total)
        return 2.* (llval1-llval0), llval1, res1, llval0, res0
    def confidence_interval(self,llval_best,extra_args={},guess={},parameter_name = "signal_rate_multiplier",two_sided = True):

        #the confidence interval computation looks in a bounded region-- we will say that we will not look for larger than 300 signal events 
        rate_multiplier_max = 10000. / self.get_mus(**extra_args).get( parameter_name.replace("_rate_multiplier",""),1.)
        rate_multiplier_min = 0.

        dl = -1*np.inf
        ul = one_parameter_interval(self.ll, parameter_name,
                rate_multiplier_max,bestfit_routine = bestfit_scipy,
                minimize_kwargs = minimize_kwargs,
                t_ppf=self.limit_threshold_function, 
                guess = guess,**extra_args)
        if two_sided: 
            extra_args_null = deepcopy(extra_args)
            extra_args_null[parameter_name] = rate_multiplier_min

            res_null, llval_null = bestfit_scipy(self.ll, guess=guess, minimize_kwargs=minimize_kwargs, **extra_args_null)
            llr =  2.*(llval_best - llval_null)
            if llr <= self.limit_threshold_function(rate_multiplier_min,0):
                dl = rate_multiplier_min
            else:
                dl = one_parameter_interval(self.ll, parameter_name,
                    rate_multiplier_min,
                    kind = "lower",
                    bestfit_routine = bestfit_scipy,
                    minimize_kwargs = minimize_kwargs,
                    t_ppf=self.limit_threshold_function, 
                    guess = guess,**extra_args)
        return dl, ul

    def toy_simulation(self,generate_args={},
            extra_args=[{},{"signal_rate_multiplier":0.}],guess={"signal_rate_multiplier":0.},compute_confidence_interval= False, confidence_interval_args = {},propagate_guess=True):
        if self.toydata_mode == "read":
            datas = self.datasets_array[self.toydata_index]
            self.toydata_index +=1
            self.assign_data(datas)
            #print("ancillary measurement is", datas[-2], datas[-2].dtype)
            #print("ancillary measurement length",len(datas[-2]))
            ancillary_measurements = structured_array_to_dict(datas[-2])
            #print("ancillary measurement",ancillary_measurements)
            self.assign_measurements(ancillary_measurements)
        else:
            datas = self.simulate_and_assign_data(generate_args=generate_args)
            ancillary_measurements =self.simulate_and_assign_measurements(generate_args=generate_args)
            if self.toydata_mode =="write":
                datas.append(dict_to_structured_array(ancillary_measurements))
                if 0<len(generate_args):
                    datas.append(dict_to_structured_array(generate_args))
                else:
                    datas.append(dict_to_structured_array({"alldefault":0}))

                self.datasets_array.append(datas)
        self.ll = LogLikelihoodSum(self.lls)
        ress = []
        extra_args_runs = extra_args
        if type(extra_args_runs) is dict:
            extra_args_runs = [extra_args_runs]

        previous_fit = {}
        for extra_args_run in extra_args_runs:
            guess_run = {}
            if propagate_guess:
                guess_run.update(previous_fit)
            guess_run.update(guess)
            res, llval = bestfit_scipy(self.ll, guess=guess_run, minimize_kwargs=minimize_kwargs, **extra_args_run)
            res.update(extra_args_run)
            res["ll"] = llval
            res["dl"] = -1.
            res["ul"] = -1.
            ress.append(res)

        if compute_confidence_interval:
            ci_guess = deepcopy(ress[-1])
            ci_guess.pop("signal_rate_multiplier",None)
            ci_args = {"llval_best":ress[-1]["ll"], "extra_args":extra_args_runs[-1],"guess":ci_guess}
            ci_args.update(confidence_interval_args)
            dl, ul = self.confidence_interval(**ci_args)
            ress[-1]["dl"] = dl
            ress[-1]["ul"] = ul



        return ress
    def get_mus(self,**res):
        ret = {}
        for ll in self.lls:
            mus = ll(full_output=True,**res)[1]
            for n,mu in zip(ll.source_name_list, mus):
                ret[n] = ret.get(n,0)+mu
        return ret
    def get_parameter_list(self):
        ret = [n + "_rate_multiplier" for n in list(self.ll.rate_parameters.keys())]
        ret += list(self.ll.shape_parameters.keys())
        ret +=["ll", "dl",  "ul"]
        return ret

    def write_toydata(self):
        toydata_to_file(self.toydata_file, datasets_array = self.datasets_array, dataset_names = self.dataset_names, overwrite_existing_file=True)






        

