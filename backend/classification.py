import enum
from backend.logger import Logger
from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List
from backend.entity import Costume
from sklearn import svm
import math
import qiskit
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
from qiskit.aqua.components.multiclass_extensions import ErrorCorrectingCode, OneAgainstRest, AllPairs
from sklearn.neural_network import MLPClassifier
from ast import literal_eval as make_tuple

import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

"""
Enum for Classifications
"""


class ClassificationTypes(enum.Enum):
    classicSklearnSVM = 0  # classical implementation of SVMs in scikit learn module
    qkeQiskitSVM = 1  # quantum kernel estimation method QSVM
    variationalQiskitSVM = 2  # variational SVM method
    classicSklearnNN = 3 # classical implementation based on neutral networks (from sklearn)
    HybridQNN = 4

    @staticmethod
    def get_name(classificationType) -> str:
        name = ""
        if classificationType == ClassificationTypes.classicSklearnSVM :
            name = "classicSklearnSVM"
        elif classificationType == ClassificationTypes.qkeQiskitSVM :
            name = "qkeQiskitSVM"
        elif classificationType == ClassificationTypes.variationalQiskitSVM:
            name = "variationalQiskitSVM"
        elif classificationType == ClassificationTypes.classicSklearnNN:
            name = "classicSklearnNN"
        elif classificationType == ClassificationTypes.HybridQNN:
            name = "HybridQNN"
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
        if type == ClassificationTypes.classicSklearnNN:
            return ClassicSklearnNN()
        if type == ClassificationTypes.HybridQNN:
            return HybridQNN()
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
        ibmq_custom_backend = "",
        featuremap="ZFeatureMap",
        entanglement="linear",
        reps=2,
        shots=1024,
        multiclass_ext = "binary" # i.e. standard binary
    ):
        self.__featuremap = featuremap
        self.__backend = backend
        self.__entanglement = entanglement
        self.__reps = reps
        self.__shots = shots
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        self.__multiclass_extension = multiclass_ext
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        """ set backend: Code duplicated from clustering """  # TODO: separate from clustering & classification
        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)
        multiclass_ext = self.instanciate_multiclas_extension()

        qsvm = QSVM(feature_map, multiclass_extension=multiclass_ext)
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)
        qsvm.train(position_matrix, labels, quantum_instance)

#         kernel_matrix = qsvm.construct_kernel_matrix(x1_vec=position_matrix, quantum_instance=quantum_instance)
#         print(kernel_matrix)

        return qsvm.predict, \
            qsvm.ret['svm']['support_vectors'] if self.__multiclass_extension == "binary" else [] # the support vectors if applicable

    def instanciate_featuremap(self, feature_dimension):
        if self.__featuremap == "ZFeatureMap":
            return ZFeatureMap(feature_dimension=feature_dimension, reps=self.__reps)  # , entanglement=entanglement)
        elif self.__featuremap == "ZZFeatureMap":
            return ZZFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        elif self.__featuremap == "PauliFeatureMap":
            return PauliFeatureMap(feature_dimension=feature_dimension, entanglement=self.__entanglement, reps=self.__reps)
        else:
            Logger.error("No such feature map available: {}".format(self.__featuremap))

    def instanciate_multiclas_extension(self):
        if self.__multiclass_extension == "allPairs":
            return AllPairs()
        elif self.__multiclass_extension == "oneAgainstRest":
            return OneAgainstRest()
        elif self.__multiclass_extension == "errorCorrectingCode":
            return ErrorCorrectingCode()
        return None

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

    def get_ibmq_custom_backend(self):
        return self.__ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.__ibmq_custom_backend = ibmq_custom_backend
        return

    def get_multiclassext(self):
        return self.__multiclass_extension

    def set_multiclassext(self, multiclass_ext):
        self.__multiclass_extension = multiclass_ext

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

        parameter_backend = self.get_backend().value
        description_backend = "Backend : Enum default(aer_statevector_simulator)\n"\
            +" A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmqtoken = self.get_ibmq_token()
        description_ibmqtoken = "IBMQ-Token : str, (default='')\n"\
                            +"IBMQ-Token for access to IBMQ online service"
        params.append(("ibmqtoken", "IBMQ-Token" , description_ibmqtoken, parameter_ibmqtoken, "text", "", ""))

        parameter_multiclassext = self.get_multiclassext()
        description_multiclassext = "Multiclass extension : {'allPairs', 'oneAgainstRest', 'errorCorrectingCode', 'binary'} (default='binary')\n "\
                            +"If number of classes is greater than 2 then a multiclass scheme "\
                            +"must be supplied, in the form of a multiclass extension."
        params.append(("multiclassext", "Multiclass extension" , description_multiclassext, parameter_multiclassext, "select", ["allPairs", "oneAgainstRest", "errorCorrectingCode", "binary"]))

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
            if param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            if param[0] == "multiclassext":
                self.set_multiclassext(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class variationalQiskitSVM(Classification):

    def __init__(
        self,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend = "",
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
        self.__ibmq_custom_backend = ibmq_custom_backend
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
        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)
        optimizer = self.instanciate_optimizer()
        var_form = self.instanciate_veriational_form(dimension)

        vqc = VQC(feature_map=feature_map, optimizer=optimizer, \
                  training_dataset=get_dict_dataset(position_matrix, labels), \
                  var_form=var_form)
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)

        vqc.train(position_matrix, labels, quantum_instance)

        pred_wrapper = self.prediction_wrapper(vqc.predict)
        #print(pred_wrapper.predict(data=position_matrix))
        return pred_wrapper.predict, []

    """ this wrapper takes the prediction function and strips off unnecessary
        data from the returned results, so it fits the form of other prediction
        functions.
    """
    class prediction_wrapper():

        def __init__(self, pred_func):
            self.pred_func = pred_func

        def predict(self, data):
            result = self.pred_func(data)
            labels = result[1]
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

    def get_ibmq_custom_backend(self):
        return self.__ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.__ibmq_custom_backend = ibmq_custom_backend
        return

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

        parameter_backend = self.get_backend().value
        description_backend = "Backend : Enum default(aer_statevector_simulator)\n"\
            +" A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmqtoken = self.get_ibmq_token()
        description_ibmqtoken = "IBMQ-Token : str, (default='')\n"\
                            +"IBMQ-Token for access to IBMQ online service"
        params.append(("ibmqtoken", "IBMQ-Token" , description_ibmqtoken, parameter_ibmqtoken, "text", "", ""))

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
            if param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class ClassicSklearnNN(Classification):

    def __init__(
        self,
        alpha=1e-4,
        solver="adam",
        hidden_layer_sizes=(100,),
    ):
        self.__solver = solver
        self.__alpha = alpha
        self.__hiddenlayersizes = hidden_layer_sizes
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        classifier = MLPClassifier(alpha=self.__alpha, hidden_layer_sizes=self.__hiddenlayersizes, solver=self.__solver)
        classifier.fit(position_matrix, labels)

        return classifier.predict, []

    # getter and setter params
    def get_solver(self):
        return self.__solver

    def set_solver(self, solver):
        self.__solver = solver
        return

    def get_alpha(self):
        return self.__alpha

    def set_alpha(self, alpha):
        self.__alpha = alpha

    def get_hiddenlayersizes(self):
        return self.__hiddenlayersizes

    def set_hiddenlayersizes(self, hiddenlayersizes):
        self.__hiddenlayersizes = hiddenlayersizes

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        classificationTypeName = "Classical SVM (sklearn)"
        params.append(("name", "ClassificationType" , "Name of choosen classification type", classificationTypeName , "header"))

        parameter_solver = self.get_solver()
        description_solver = "solver : {'lbfgs', 'sgd', 'adam'}, (default='adam')\n"\
                            +"The solver for weight optimization."
        params.append(("solver", "Solver" , description_solver, parameter_solver, "select",
                       ['lbfgs', 'sgd', 'adam']))

        parameter_alpha = self.get_alpha()
        description_alpha = "alpha : float, (default=1e-4)\n"\
                            +"L2 penalty (regularization term) parameter."
        params.append(("alpha", "Alpha" , description_alpha, parameter_alpha, "number", 0, 0.0001))

        parameter_hiddenlayersizes = str(self.get_hiddenlayersizes())
        description_hiddenlayersizes = "Hidden layer sizes : tuple, (default=(100,))\n"\
                            +"The ith element represents the number of neurons in the ith hidden layer."
        params.append(("hiddenlayersizes", "Hidden layer sizes" , description_hiddenlayersizes, parameter_hiddenlayersizes, "text", "", ""))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "solver":
                self.set_solver(param[3])
            if param[0] == "alpha":
                self.set_alpha(param[3])
            if param[0] == "hiddenlayersizes":
                self.set_hiddenlayersizes(make_tuple(param[3]))

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class HybridQNN(Classification):
    def __init__(
        self,
        backend=QuantumBackends.aer_qasm_simulator,
        ibmq_token="",
        ibmq_custom_backend = "",
        shots = 1024,
        epochs = 10,
        learning_rate = 0.01,
        batch_size = 1,
        shuffle = True
    ):
        self.__backend = backend
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        self.__shuffle = shuffle
        self.__shots = shots
        self.__epochs = epochs
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        n_samples = len(labels)
        n_classes = len(list(set(labels)))
        if n_classes > 2:
            raise Exception("Multi-class support for "+ str(type(self)) +" not implemented, yet.")

        # shuffle
        if self.__shuffle:
            zipped = list(zip(position_matrix, labels))
            random.shuffle(zipped)
            position_matrix, labels = zip(*zipped)

        # Conversion of labels to int necessary for tensorization
        position_matrix, labels = list(position_matrix), [int(i) for i in labels]

        # convert labels to one-hot
        #labels = [np.eye(n_classes).astype(int)[label] for label in labels]

        # tensorize and floatify
        position_matrix = torch.tensor([item for item in position_matrix]).float()
        labels = torch.tensor(labels)
        labels = torch.tensor([item for item in labels])
        labels = labels.long()

        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)
        model = Net(backend, self.__shots)
        optimizer = optim.Adam(model.parameters(), lr=self.__learning_rate)

        loss_func = nn.NLLLoss()

        loss_list = []

        model.train()
        for epoch in range(self.__epochs):
            total_loss = []
            for batch_idx, (data, target) in enumerate(zip(chunk(position_matrix, self.__batch_size), chunk(labels, self.__batch_size))): # this implies batch size 1
                optimizer.zero_grad()
                # Forward pass
                output = model(data)
                # Calculating loss
                loss = loss_func(output, target)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer.step()

                total_loss.append(loss.item())
            loss_list.append(sum(total_loss)/len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
                100. * (epoch + 1) / self.__epochs, loss_list[-1]))
        model.eval()

        def prediction_fun(test_data):
            result = []
            for data_point in test_data:
                data_tensor = torch.tensor([data_point]).float()
                data_res = model(data_tensor)
                result.append(data_res)
            lbls = [int(torch.argmax(res)) for res in result]
            return np.array(lbls)

        return prediction_fun, []

    # getter and setter params
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

    def get_ibmq_custom_backend(self):
        return self.__ibmq_custom_backend

    def set_ibmq_custom_backend(self, ibmq_custom_backend):
        self.__ibmq_custom_backend = ibmq_custom_backend
        return

    def get_epochs(self):
        return self.__epochs

    def set_epochs(self, epochs):
        self.__epochs = epochs

    def get_learningrate(self):
        return self.__learning_rate

    def set_learningrate(self, learningrate):
        self.__learning_rate = learningrate

    def get_batchsize(self):
        return self.__batch_size

    def set_batchsize(self, batch_size):
        self.__batch_size = batch_size

    def get_shuffle(self):
        return self.__shuffle

    def set_shuffle(self, shuffle):
        self.__shuffle = shuffle

    def get_param_list(self) -> list:
        """
        # each tuple has informations as follows
        # (pc_name[0] , showedname[1] , description[2] , actual value[3] , input type[4] ,
        # [5] number(min steps)/select (options) /checkbox() / text )
        """
        params = []
        classificationTypeName = "Hybrid quantum-classical neural network"
        params.append(("name", "ClassificationType" , "Name of choosen classification type", classificationTypeName , "header"))

        parameter_backend = self.get_backend().value
        description_backend = "Backend : Enum default(aer_statevector_simulator)\n"\
            +" A list of possible backends. aer is a local simulator and ibmq are backends provided by IBM."
        params.append(("quantumBackend", "QuantumBackend", description_backend, parameter_backend, "select", [qb.value for qb in QuantumBackends]))

        parameter_ibmq_custom_backend = self.get_ibmq_custom_backend()
        description_ibmq_custom_backend = "str default(\"\") "\
            + " The name of a custom backend of ibmq."
        params.append(("ibmqCustomBackend", "IBMQ-Custom-Backend", description_ibmq_custom_backend, str(parameter_ibmq_custom_backend), "text", "", ""))

        parameter_ibmqtoken = self.get_ibmq_token()
        description_ibmqtoken = "IBMQ-Token : str, (default='')\n"\
                            +"IBMQ-Token for access to IBMQ online service"
        params.append(("ibmqtoken", "IBMQ-Token" , description_ibmqtoken, parameter_ibmqtoken, "text", "", ""))

        parameter_shots = self.get_shots()
        description_shots = "Shots : int, (default=1024)\n"\
                            +"Number of repetitions of each circuit, for sampling."
        params.append(("shots", "Shots" , description_shots, parameter_shots, "number", 1, 1))

        parameter_epochs = self.get_epochs()
        description_epochs = "Epochs : int, (default=10)\n"\
                            +"Number of learning epochs."
        params.append(("epochs", "Epochs" , description_epochs, parameter_epochs, "number", 1, 1))

        parameter_learningrate = self.get_learningrate()
        description_learningrate = "Learning rate parameter"
        params.append(("learningrate", "Learning rate", description_learningrate, parameter_learningrate, "number", 0, 1e-3))

        parameter_batchsize = self.get_batchsize()
        description_batchsize = "Batch size : int, (default=1)\n"\
                            +"Batch size for training"
        params.append(("batchsize", "Batchsize" , description_batchsize, parameter_batchsize, "number", 1, 1))

        parameter_shuffle = self.get_shuffle()
        description_shuffle = "shuffle: bool, (default=True)\n If True: randomly shuffle data before training"
        params.append(("shuffle", "Shuffle" , description_shuffle, parameter_shuffle, "checkbox"))

        return params

    def set_param_list(self, params: list=[]) -> np.matrix:
        for param in params:
            if param[0] == "shots":
                self.set_shots(param[3])
            if param[0] == "ibmqtoken":
                self.set_ibmq_token(param[3])
            if param[0] == "quantumBackend":
                self.set_backend(QuantumBackends[param[3]])
            if param[0] == "ibmqCustomBackend":
                self.set_ibmq_custom_backend(param[3])
            if param[0] == "epochs":
                self.set_epochs(param[3])
            if param[0] == "learningrate":
                self.set_learningrate(param[3])
            if param[0] == "batchsize":
                self.set_batchsize(param[3])
            if param[0] == "shuffle":
                self.set_shuffle(param[3])


    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass

# source https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
# Some classes for the hybrid neural network approach
class NNQuantumCircuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots = self.shots,
                             parameter_binds = [{self.theta: theta} for theta in thetas])
        result = job.result().get_counts(self._circuit)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])

    def draw_circuit(self):
        self ._circuit.draw()

class HybridFunction1(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = [ctx.quantum_circuit.run(input[i].tolist()) for i in range(len(input))]
        result = torch.tensor(expectation_z)
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left  = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient)
        gradients = np.array([gradients]).T

        result_gradient = (torch.tensor([gradients]).float() * grad_output.float())

        return result_gradient.view(len(input), 1), None, None

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = NNQuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction1.apply(input, self.quantum_circuit, self.shift)

class Net(nn.Module):
    def __init__(self, backend, shots):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.hybrid = Hybrid(backend, shots, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)

def chunk(data, batch_size):
    chunks = []
    for i in range(0, len(data), batch_size):
        chunks.append(data[i:i+batch_size])
    return chunks

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
