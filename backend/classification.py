import enum
from backend.logger import Logger
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
from backend.entity import Costume
from sklearn import svm
import math
from qiskit import Aer
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.quantum_instance import QuantumInstance
from backend.clustering import QuantumBackends  # TODO: separate from clustering
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap

from qiskit import IBMQ
from qiskit.aqua.algorithms.classifiers.vqc import VQC
from qiskit.aqua.components.optimizers import ADAM, AQGD, BOBYQA, COBYLA, NELDER_MEAD, SPSA, POWELL, NFT, TNC
                                        # some more https://qiskit.org/documentation/apidoc/qiskit.aqua.components.optimizers.html
import sys
from qiskit.circuit.library.n_local import RealAmplitudes, ExcitationPreserving, EfficientSU2
from qiskit.circuit.library.n_local.two_local import TwoLocal
import random
from numpy import setdiff1d

"""
Enum for Classifications
"""


class ClassificationTypes(enum.Enum):
    classicSklearnSVM = 0  # classical implementation of SVMs in scikit learn module
    qkeQiskitSVM = 1  # quantum kernel estimation method QSVM
    variationalQiskitSVM = 2  # variational SVM method

    @staticmethod
    def get_name(classificationType) -> str:
        name = ""
        if classificationType == ClassificationTypes.classicSklearnSVM :
            name = "classicSklearnSVM"
        elif classificationType == ClassificationTypes.qkeQiskitSVM :
            name = "qkeQiskitSVM"
        elif classificationType == ClassificationTypes.variationalQiskitSVM:
            name = "variationalQiskitSVM"
        else:
            Logger.error("No name for classification \"" + str(classificationType))
            raise ValueError("No name for classification \"" + str(classificationType))
        return name

    """
    Returns the description of the given ClassificationType.
    """

    @staticmethod
    def get_description(classificationType) -> str:
        description = ""
        if classificationType == ClassificationTypes.classicSklearnSVM:
            description = ("Implementation of SVM as provided by sklearn")
        else:
            Logger.error("No description for classification \"" + str(classificationType) + "\" specified")
            raise ValueError("No description for classification \"" + str(classificationType) + "\" specified")
        return description


class Classification(metaclass=ABCMeta):
    """
    Interface for Classification Object
    """

    @abstractmethod
    def create_classifier(self, position_matrix : np.matrix , labels: list, similarity_matrix : np.matrix) -> np.matrix:
        pass

    @abstractmethod
    def get_param_list(self) -> list:
        pass

    @abstractmethod
    def set_param_list(self, params: list=[]) -> np.matrix:
        pass

    @abstractmethod
    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

"""
Represents the factory to create an classification object
"""


class ClassificationFactory:

    @staticmethod
    def create(type: ClassificationTypes) -> Classification:
        if type == ClassificationTypes.classicSklearnSVM:
            return ClassicSklearnSVM()
        if type == ClassificationTypes.qkeQiskitSVM:
            return qkeQiskitSVM()
        if type == ClassificationTypes.variationalQiskitSVM:
            return variationalQiskitSVM()
        else:
            Logger.error("Unknown type of clustering. The application will quit know.")
            raise Exception("Unknown type of clustering.")


class ClassicSklearnSVM(Classification):

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
    ):
        self.__kernel = kernel
        self.__C_param = C
        self.__degree = degree
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        classifier = svm.SVC(C=self.__C_param, kernel=self.__kernel, degree=self.__degree)
        classifier.fit(position_matrix, labels)

        return classifier.predict, classifier.support_vectors_

    # getter and setter params
    def get_kernel(self):
        return self.__kernel

    def set_kernel(self, kernel):
        self.__kernel = kernel
        return

    def get_C_param(self):
        return self.__C_param

    def set_C_para(self, C_param):
        self.__C_param = C_param

    def get_degree(self):
        return self.__degree

    def set_degree(self, degree):
        self.__degree = degree

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        classificationTypeName = "Classical SVM (sklearn)"
        params.append(("name", "ClassificationType" , "Name of choosen classification type", classificationTypeName , "header"))

        parameter_kernel = self.get_kernel()
        description_kernel = "kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, (default='rbf')\n"\
                            +"Specifies the kernel type to be used in the algorithm. 'precomputed' not selectable here."
        params.append(("kernel", "Kernel" , description_kernel, parameter_kernel, "select",
                       ["linear", "poly", "rbf", "sigmoid"]))

        parameter_C = self.get_C_param()
        description_C = "C : float, (default=1.0)\n"\
                            +"Specifies the kernel type to be used in the algorithm. 'precomputed' not selectable here."
        params.append(("C", "C" , description_C, parameter_C, "number", 0, 0.001))

        parameter_degree = self.get_degree()
        description_degree = "degree : int, (default=3)\n"\
                            +"Degree of the polynomial kernel function ('poly'). Ignored by all other kernels."
        params.append(("degree", "Degree" , description_degree, parameter_degree, "number", 1, 1))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "kernel":
                self.set_kernel(param[3])
            if param[0] == "C":
                self.set_C_para(param[3])
            if param[0] == "degree":
                self.set_degree(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class qkeQiskitSVM(Classification):

    def __init__(
        self,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        featuremap="ZFeatureMap",
        entanglement="linear",
        reps=2,
        shots=1024
    ):
        self.__featuremap = featuremap
        self.__backend = backend
        self.__entanglement = entanglement
        self.__reps = reps
        self.__shots = shots
        self.__ibmq_token = ibmq_token
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        labels = np.where(labels==-1, 0, labels) # relabeling

        """ set backend: Code duplicated from clustering """  # TODO: separate from clustering & classification
        if self.__backend.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.__backend.name[4:]
            backend = Aer.get_backend(aer_backend_name)
        elif self.__backend.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(self.__ibmq_token)
            backend = provider.get_backend(self.__backend.value)
        else:
            Logger.error("Unknown quantum backend specified!")

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)

        qsvm = QSVM(feature_map)
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)
        qsvm.train(position_matrix, labels, quantum_instance)

#         kernel_matrix = qsvm.construct_kernel_matrix(x1_vec=position_matrix, quantum_instance=quantum_instance)
#         print(kernel_matrix)

        pred_wrapper = self.prediction_wrapper(qsvm.predict)

        #print(pred_wrapper.predict(data=position_matrix))
        return pred_wrapper.predict, qsvm.ret['svm']['support_vectors']

    """ this wrapper replaces labels 0 by -1.
    """
    class prediction_wrapper():

        def __init__(self, pred_func):
            self.pred_func = pred_func

        def predict(self, data):
            result = self.pred_func(data)
            labels = np.array(result)#[1]
            labels = np.where(labels==0, -1, labels)
            return labels

    def instanciate_featuremap(self, feature_dimension):
        if self.__featuremap == "ZFeatureMap":
            return ZFeatureMap(feature_dimension=feature_dimension, reps=self.__reps)  # , entanglement=entanglement)
        elif self.__featuremap == "ZZFeatureMap":
            return ZZFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        elif self.__featuremap == "PauliFeatureMap":
            return PauliFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        else:
            Logger.error("No such feature map available: {}".format(self.__featuremap))

    # getter and setter params
    def get_featuremap(self):
        return self.__featuremap

    def set_featuremap(self, featuremap):
        self.__featuremap = featuremap
        return

    def get_entanglement(self):
        return self.__entanglement

    def set_entanglement(self, entanglement):
        self.__entanglement = entanglement

    def get_shots(self):
        return self.__shots

    def set_shots(self, shots):
        self.__shots = shots

    def get_backend(self):
        return self.__backend

    def set_backend(self, backend):
        self.__backend = backend

    def get_ibmq_token(self):
        return self.__ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.__ibmq_token = ibmq_token

    def get_reps(self):
        return self.__reps

    def set_reps(self, reps):
        self.__reps = reps

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        classificationTypeName = "QKE SVM (using qiskit)"
        params.append(("name", "ClassificationType" , "Name of choosen classification type", classificationTypeName , "header"))

        parameter_featuremap = self.get_featuremap()
        description_featuremap = "Feature Map : {'ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap'}, (default='ZFeatureMap')\n"\
                            +"Feature map module used to transform data."
        params.append(("featuremap", "Feature Map" , description_featuremap, parameter_featuremap, "select",
                       ["ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"]))

        parameter_entanglement = self.get_entanglement()
        description_entanglement = "Entanglement : {'full', 'linear'}, (default='full')\n"\
                            +"Specifies the entanglement structure."
        params.append(("entanglement", "Entanglement" , description_entanglement, parameter_entanglement, "select",
                       ["full", "linear"]))

        parameter_reps = self.get_reps()
        description_reps = "reps : int, (default=2)\n"\
                            +"The number of repeated circuits."
        params.append(("reps", "Repetitions" , description_reps, parameter_reps, "number", 1, 1))

        parameter_shots = self.get_shots()
        description_shots = "Shots : int, (default=1024)\n"\
                            +"Number of repetitions of each circuit, for sampling."
        params.append(("shots", "Shots" , description_shots, parameter_shots, "number", 1, 1))

        parameter_ibmqtoken = self.get_ibmq_token()
        description_ibmqtoken = "IBMQ-Token : str, (default='')\n"\
                            +"IBMQ-Token for access to IBMQ online service"
        params.append(("ibmqtoken", "IBMQ-Token" , description_ibmqtoken, parameter_ibmqtoken, "text", "", ""))

        parameter_backend = self.get_backend().value
        description_backend = "Backend : Enum default(aer_statevector_simulator)\n"\
            +" A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "featuremap":
                self.set_featuremap(param[3])
            if param[0] == "entanglement":
                self.set_entanglement(param[3])
            if param[0] == "reps":
                self.set_reps(param[3])
            if param[0] == "shots":
                self.set_shots(param[3])
            if param[0] == "ibmqtoken":
                self.set_ibmq_token(param[3])
            if param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class variationalQiskitSVM(Classification):

    def __init__(
        self,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        featuremap="ZFeatureMap",
        entanglement="linear",
        reps=2,
        shots=1024,
        var_form="RyRz",
        reps_varform = 3,
        optimizer="SPSA",
        maxiter=100,
        adam_tolerance = 0,
        adam_learningrate = 0.1,
        adam_noicefactor = 1e-8,
        adam_epsilon = 1e-10,
    ):
        self.__featuremap = featuremap
        self.__backend = backend
        self.__entanglement = entanglement
        self.__reps = reps
        self.__shots = shots
        self.__ibmq_token = ibmq_token
        self.__var_form = var_form
        self.__reps_varform = reps_varform
        self.__optimizer = optimizer
        self.__maxiter = maxiter
        self.__adam_tolerance = adam_tolerance
        self.__adam_learningrate = adam_learningrate
        self.__adam_noisefactor = adam_noicefactor
        self.__adam_epsilon = adam_epsilon
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        labels = np.where(labels==-1, 0, labels) # relabeling

        """ set backend: Code duplicated from clustering """  # TODO: separate from clustering & classification
        backend = None
        if self.__backend.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.__backend.name[4:]
            backend = Aer.get_backend(aer_backend_name)
        elif self.__backend.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(self.__ibmq_token)
            backend = provider.get_backend(self.__backend.value)
        else:
            Logger.error("Unknown quantum backend specified!")

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)
        optimizer = self.instanciate_optimizer()
        var_form = self.instanciate_veriational_form(dimension)

        vqc = VQC(feature_map=feature_map, optimizer=optimizer, training_dataset=get_dict_dataset(position_matrix, labels), var_form=var_form)
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)

        vqc.train(position_matrix, labels, quantum_instance)

        pred_wrapper = self.prediction_wrapper(vqc.predict)
        #print(pred_wrapper.predict(data=position_matrix))
        return pred_wrapper.predict, []

    """ this wrapper takes the prediction function and strips off unnecessary
        data from the returned results, so it fits the form of other prediction
        functions. It also replaces labels 0 by -1.
    """
    class prediction_wrapper():

        def __init__(self, pred_func):
            self.pred_func = pred_func

        def predict(self, data):
            result = self.pred_func(data)
            labels = result[1]
            labels = np.where(labels==0, -1, labels)
            return labels

    def instanciate_featuremap(self, feature_dimension):
        if self.__featuremap == "ZFeatureMap":
            return ZFeatureMap(feature_dimension=feature_dimension, reps=self.__reps)  # , entanglement=entanglement)
        elif self.__featuremap == "ZZFeatureMap":
            return ZZFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        elif self.__featuremap == "PauliFeatureMap":
            return PauliFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        else:
            Logger.error("No such feature map available: {}".format(self.__featuremap))

    def instanciate_veriational_form(self, feature_dimension):
        if self.__var_form == "RealAmplitudes":
            return RealAmplitudes(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__var_form == "ExcitationPreserving":
            return ExcitationPreserving(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__var_form == "EfficientSU2":
            return EfficientSU2(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__var_form == "RyRz":
            return TwoLocal(num_qubits=feature_dimension, rotation_blocks=['ry', 'rz'], entanglement_blocks="cz", entanglement=self.__entanglement, reps=3)
        else:
            Logger.error("No such variational form available: {}".format(self.__var_form))

    def instanciate_optimizer(self):
        if self.__optimizer == "ADAM":
            return ADAM(maxiter=self.__maxiter, tol = self.__adam_tolerance, lr=self.__adam_learningrate,
                        noise_factor=self.__adam_noisefactor, eps=self.__adam_epsilon)
        if self.__optimizer == "AQGD":
            return AQGD(maxiter=self.__maxiter)
        if self.__optimizer == "BOBYQA":
            return BOBYQA(maxiter=self.__maxiter)
        if self.__optimizer == "COBYLA":
            return COBYLA(maxiter=self.__maxiter)
        if self.__optimizer == "NELDER_MEAD":
            return NELDER_MEAD(maxiter=self.__maxiter)
        if self.__optimizer == "SPSA":
            return SPSA(maxiter=self.__maxiter)
        if self.__optimizer == "POWELL":
            return POWELL(maxiter=self.__maxiter)
        if self.__optimizer == "NFT":
            return NFT(maxiter=self.__maxiter)
        if self.__optimizer == "TNC":
            return TNC(maxiter=self.__maxiter)

    # getter and setter params
    def get_featuremap(self):
        return self.__featuremap

    def set_featuremap(self, featuremap):
        self.__featuremap = featuremap
        return

    def get_entanglement(self):
        return self.__entanglement

    def set_entanglement(self, entanglement):
        self.__entanglement = entanglement

    def get_shots(self):
        return self.__shots

    def set_shots(self, shots):
        self.__shots = shots

    def get_backend(self):
        return self.__backend

    def set_backend(self, backend):
        self.__backend = backend

    def get_ibmq_token(self):
        return self.__ibmq_token

    def set_ibmq_token(self, ibmq_token):
        self.__ibmq_token = ibmq_token

    def get_reps(self):
        return self.__reps

    def set_reps(self, reps):
        self.__reps = reps

    def get_var_form(self):
        return self.__var_form

    def set_var_form(self, var_form):
        self.__var_form = var_form

    def get_repsvarform(self):
        return self.__reps_varform

    def set_repsvarform(self, reps):
        self.__reps_varform = reps

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def get_maxiter(self):
        return self.__maxiter

    def set_maxiter(self, maxiter):
        self.__maxiter = maxiter

    def get_adamtolerance(self):
        return self.__adam_tolerance

    def get_adamlearningrate(self):
        return self.__adam_learningrate

    def get_adamnoisefactor(self):
        return self.__adam_noisefactor

    def get_adamepsilon(self):
        return self.__adam_epsilon

    def set_adamtolerance(self, adam_tolerance):
        self.__adam_tolerance = adam_tolerance

    def set_adamlearningrate(self, adam_learningrate):
        self.__adam_learningrate = adam_learningrate

    def set_adamnoisefactor(self, adam_noicefactor):
        self.__adam_noisefactor = adam_noicefactor

    def set_adamepsilon(self, adam_epsilon):
        self.__adam_epsilon = adam_epsilon

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        classificationTypeName = "variational SVM (using qiskit)"
        params.append(("name", "ClassificationType" , "Name of choosen classification type", classificationTypeName , "header"))

        parameter_featuremap = self.get_featuremap()
        description_featuremap = "Feature Map : {'ZFeatureMap', 'ZZFeatureMap', 'PauliFeatureMap'}, (default='ZFeatureMap')\n"\
                            +"Feature map module used to transform data."
        params.append(("featuremap", "Feature Map" , description_featuremap, parameter_featuremap, "select",
                       ["ZFeatureMap", "ZZFeatureMap", "PauliFeatureMap"]))

        parameter_entanglement = self.get_entanglement()
        description_entanglement = "Entanglement : {'full', 'linear'}, (default='full')\n"\
                            +"Specifies the entanglement structure."
        params.append(("entanglement", "Entanglement" , description_entanglement, parameter_entanglement, "select",
                       ["full", "linear"]))

        parameter_reps = self.get_reps()
        description_reps = "reps : int, (default=2)\n"\
                            +"The number of repeated circuits."
        params.append(("reps", "Repetitions" , description_reps, parameter_reps, "number", 1, 1))

        parameter_shots = self.get_shots()
        description_shots = "Shots : int, (default=1024)\n"\
                            +"Number of repetitions of each circuit, for sampling."
        params.append(("shots", "Shots" , description_shots, parameter_shots, "number", 1, 1))

        parameter_ibmqtoken = self.get_ibmq_token()
        description_ibmqtoken = "IBMQ-Token : str, (default='')\n"\
                            +"IBMQ-Token for access to IBMQ online service"
        params.append(("ibmqtoken", "IBMQ-Token" , description_ibmqtoken, parameter_ibmqtoken, "text", "", ""))

        parameter_backend = self.get_backend().value
        description_backend = "Backend : Enum default(aer_statevector_simulator)\n"\
            +" A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_varform = self.get_var_form()
        description_varform = "Variational Form : {'RealAmplitudes', 'ExcitationPreserving', 'EfficientSU2', 'RyRz'}, (default='RyRz')\n"\
                                +"The variational form instance."
        params.append(("varform", "Variational Form", description_varform, parameter_varform, "select", ["RealAmplitudes", "ExcitationPreserving", "EfficientSU2", "RyRz"]))

        parameter_repsvarform = self.get_repsvarform()
        description_repsvarform= "reps: int (default=3)\n For variational form;"\
                                    +"Specifies how often a block consisting of a rotation layer and entanglement"\
                                    +"layer is repeated."
        params.append(("reps_varform", "Repetitions (variational Form)", description_repsvarform, parameter_repsvarform, "number", 1,1))

        parameter_optimizer = self.get_optimizer()
        description_optimizer = "Optimizer : {'ADAM', 'AQGD', 'BOBYQA', 'COBYLA', 'NELDER_MEAD', 'SPSA', 'POWELL', 'NFT', 'TNC'}, (default='SPSA')\n"\
                                    +"The classical optimizer to use."
        params.append(("optimizer", "Optimizer", description_optimizer, parameter_optimizer, "select", ["ADAM", "AQGD", "BOBYQA", "COBYLA", "NELDER_MEAD", "SPSA", "POWELL", "NFT", "TNC"]))

        parameter_maxiter = self.get_maxiter()
        description_maxiter = "Max iterations : int (default=100)\n For optimizer;"\
                                +"Maximum number of iterations to perform."
        params.append(("maxiter", "Max iterations", description_maxiter, parameter_maxiter, "number", 1, 1))

        parameter_adamtolerance = self.get_adamtolerance()
        description_adamtolerance = "Tolerance parameter of ADAM optimizer"
        params.append(("adam_tolerance", "ADAM Optimizer: Tolerance", description_adamtolerance, parameter_adamtolerance, "number", 0, 1e-6))

        parameter_adamlearningrate = self.get_adamlearningrate()
        description_adamlearningrate = "Learning rate parameter of ADAM optimizer"
        params.append(("adam_learningrate", "ADAM Optimizer: Learning rate", description_adamlearningrate, parameter_adamlearningrate, "number", 0, 1e-3))

        parameter_adamnoisefactor = self.get_adamnoisefactor()
        description_adamnoisefactor = "Noise factor parameter of ADAM optimizer"
        params.append(("adam_noise", "ADAM Optimizer: Noise factor", description_adamnoisefactor, parameter_adamnoisefactor, "number", 0, 1e-8))

        parameter_adamepsilon = self.get_adamepsilon()
        description_adamepsilon = "Epsilon parameter of ADAM optimizer"
        params.append(("adam_epsilon", "ADAM Optimizer: Epsilon", description_adamepsilon, parameter_adamepsilon, "number", 0, 1e-10))
        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "featuremap":
                self.set_featuremap(param[3])
            if param[0] == "entanglement":
                self.set_entanglement(param[3])
            if param[0] == "reps":
                self.set_reps(param[3])
            if param[0] == "shots":
                self.set_shots(param[3])
            if param[0] == "ibmqtoken":
                self.set_ibmq_token(param[3])
            if param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            if param[0] == "varform":
                self.set_var_form(param[3])
            if param[0] == "reps_varform":
                self.set_repsvarform(param[3])
            if param[0] == "optimizer":
                self.set_optimizer(param[3])
            if param[0] == "maxiter":
                self.set_maxiter(param[3])
            if param[0] == "adam_tolerance":
                self.set_adamtolerance(param[3])
            if param[0] == "adam_learningrate":
                self.set_adamlearningrate(param[3])
            if param[0] == "adam_noise":
                self.set_adamnoisefactor(param[3])
            if param[0] == "adam_epsilon":
                self.set_adamepsilon(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

""" transforms data into a dictionary of the form
    {'A': np.ndarray, 'B': np.ndarray, ...}.
    as required for VQC initialization
"""
def get_dict_dataset(position_matrix, labels):
    dict = {}
    for label in set(labels):
        dict[label] = []
    for i in range(len(position_matrix)):
        dict[labels[i]].append(position_matrix[i])
    for key in dict:
        dict[key] = np.array(dict[key])
    return dict
