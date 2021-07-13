import sys
import enum
import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod
import random
from ast import literal_eval as make_tuple

from backend.logger import Logger
from backend.entity import Costume
from backend.clustering import QuantumBackends  # TODO: separate from clustering

from sklearn import svm
from sklearn.neural_network import MLPClassifier

from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import TwoLocal, RealAmplitudes, ExcitationPreserving, EfficientSU2, ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit.algorithms.optimizers import ADAM, AQGD, BOBYQA, COBYLA, NELDER_MEAD, SPSA, POWELL, NFT, TNC

import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, SGD, Rprop, RMSprop, LBFGS
import torch.nn as nn
from backend.QNN import DressedQNN

"""
Enum for Classifications
"""


class ClassificationTypes(enum.Enum):
    classicSklearnSVM = 0  # classical implementation of SVMs in scikit learn module
    qiskitQSVC = 1  # quantum kernel estimation method QSVM
    qiskitVQC = 2  # variational SVM method
    classicSklearnNN = 3 # classical implementation based on neutral networks (from sklearn)
    HybridQNN = 4

    @staticmethod
    def get_name(classificationType) -> str:
        name = ""
        if classificationType == ClassificationTypes.classicSklearnSVM :
            name = "classicSklearnSVM"
        elif classificationType == ClassificationTypes.qiskitQSVC :
            name = "qiskitQSVC"
        elif classificationType == ClassificationTypes.qiskitVQC:
            name = "qiskitVQC"
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
        if type == ClassificationTypes.qiskitQSVC:
            return qiskitQSVC()
        if type == ClassificationTypes.qiskitVQC:
            return qiskitVQC()
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
                            +"Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty."
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


class qiskitQSVC(Classification):

    def __init__(
        self,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend = "",
        featuremap="ZFeatureMap",
        entanglement="linear",
        reps=2,
        shots=1024,
    ):
        self.__featuremap = featuremap
        self.__backend = backend
        self.__entanglement = entanglement
        self.__reps = reps
        self.__shots = shots
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        n_samples = len(labels)
        n_classes = len(list(set(labels)))
        Logger.debug("Training classifier: {}\nN samples: {}\nN classes: {}".format(str(type(self)), n_samples, n_classes))

        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)
        Logger.debug("Backend: {}".format(self.__backend))

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)
        kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
        Logger.debug("Dimensions: {}\nFeature Map: {}".format(dimension, self.__featuremap))

        Logger.debug("Initialized quantum instance.\nStart training...")
        qsvc = QSVC(quantum_kernel=kernel)
        qsvc.fit(position_matrix, labels)

        Logger.debug("Training complete.")
        return qsvc.predict, qsvc.support_vectors_

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
        classificationTypeName = "qiskit Quantum Support Vector Classifier (QSVC; using quantum kernel)"
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

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


class qiskitVQC(Classification):

    def __init__(
        self,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend = "",
        featuremap="ZFeatureMap",
        entanglement="linear",
        reps=2,
        shots=1024,
        ansatz="RyRz",
        reps_ansatz = 3,
        optimizer="SPSA",
        maxiter=100
    ):
        self.__featuremap = featuremap
        self.__backend = backend
        self.__entanglement = entanglement
        self.__reps = reps
        self.__shots = shots
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        self.__ansatz = ansatz
        self.__reps_ansatz = reps_ansatz
        self.__optimizer = optimizer
        self.__maxiter = maxiter
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        n_samples = len(labels)
        n_classes = len(list(set(labels)))
        if n_classes > 2:
            raise Exception("Multi-class support for "+ str(type(self)) +" not implemented, yet.")

        backend = QuantumBackends.get_quantum_backend(self.__backend, self.__ibmq_token, self.__ibmq_custom_backend)

        dimension = position_matrix.shape[1]
        feature_map = self.instanciate_featuremap(feature_dimension=dimension)
        ansatz = self.instanciate_ansatz(dimension)
        optimizer = self.instanciate_optimizer()
        quantum_instance = QuantumInstance(backend, seed_simulator=9283712, seed_transpiler=9283712, shots=self.__shots)

        vqc = VQC(feature_map=feature_map, ansatz=ansatz,
                  optimizer=optimizer, quantum_instance=quantum_instance)

        # convert labels to one-hot
        labels_onehot = np.zeros((n_samples, 2))
        for i in range(n_samples):
            labels_onehot[i, labels[i]] = 1

        vqc.fit(position_matrix, labels_onehot)

        pred_wrapper = self.prediction_wrapper(vqc.predict)
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
            # convert back from one-hot to class
            labels = np.array([r[1] for r in result])
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

    def instanciate_ansatz(self, feature_dimension):
        if self.__ansatz == "RealAmplitudes":
            return RealAmplitudes(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__ansatz == "ExcitationPreserving":
            return ExcitationPreserving(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__ansatz == "EfficientSU2":
            return EfficientSU2(num_qubits=feature_dimension, entanglement=self.__entanglement)
        elif self.__ansatz == "RyRz":
            return TwoLocal(num_qubits=feature_dimension, rotation_blocks=['ry', 'rz'], entanglement_blocks="cz", entanglement=self.__entanglement, reps=3)
        else:
            Logger.error("No such ansatz available: {}".format(self.__ansatz))

    def instanciate_optimizer(self):
        if self.__optimizer == "ADAM":
            return ADAM(maxiter=self.__maxiter)
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

    def get_ansatz(self):
        return self.__ansatz

    def set_ansatz(self, ansatz):
        self.__ansatz = ansatz

    def get_reps_ansatz(self):
        return self.__reps_ansatz

    def set_reps_ansatz(self, reps):
        self.__reps_ansatz = reps

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def get_maxiter(self):
        return self.__maxiter

    def set_maxiter(self, maxiter):
        self.__maxiter = maxiter

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
        classificationTypeName = "qiskit Variational Quantum Classifier (VQC)"
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

        parameter_ansatz = self.get_ansatz()
        description_ansatz = "Ansatz : {'RealAmplitudes', 'ExcitationPreserving', 'EfficientSU2', 'RyRz'}, (default='RyRz')\n"
        params.append(("ansatz", "Ansatz", description_ansatz, parameter_ansatz, "select", ["RealAmplitudes", "ExcitationPreserving", "EfficientSU2", "RyRz"]))

        parameter_reps_ansatz = self.get_reps_ansatz()
        description_reps_ansatz= "reps: int (default=3)\n For ansatz;"\
                                    +"Specifies how often a block consisting of a rotation layer and entanglement"\
                                    +"layer is repeated."
        params.append(("reps_ansatz", "Repetitions (ansatz)", description_reps_ansatz, parameter_reps_ansatz, "number", 1,1))

        parameter_optimizer = self.get_optimizer()
        description_optimizer = "Optimizer : {'ADAM', 'AQGD', 'BOBYQA', 'COBYLA', 'NELDER_MEAD', 'SPSA', 'POWELL', 'NFT', 'TNC'}, (default='SPSA')\n"\
                                    +"The classical optimizer to use."
        params.append(("optimizer", "Optimizer", description_optimizer, parameter_optimizer, "select", ["ADAM", "AQGD", "BOBYQA", "COBYLA", "NELDER_MEAD", "SPSA", "POWELL", "NFT", "TNC"]))

        parameter_maxiter = self.get_maxiter()
        description_maxiter = "Max iterations : int (default=100)\n For optimizer;"\
                                +"Maximum number of iterations to perform."
        params.append(("maxiter", "Max iterations", description_maxiter, parameter_maxiter, "number", 1, 1))

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
            if param[0] == "ansatz":
                self.set_ansatz(param[3])
            if param[0] == "reps_ansatz":
                self.set_reps_ansatz(param[3])
            if param[0] == "optimizer":
                self.set_optimizer(param[3])
            if param[0] == "maxiter":
                self.set_maxiter(param[3])
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
        optimizer = 'Adam',
        learning_rate = 0.07,
        batch_size = 10,
        shuffle = True,
        hidden_layers_preproc = (4,),
        hidden_layers_postproc = (4,),
        repetitions_qlayer = 4,
        shift = np.pi / 4,
        weight_initialization = "uniform",
        w2w = 0
    ):
        self.__backend = backend
        self.__ibmq_token = ibmq_token
        self.__ibmq_custom_backend = ibmq_custom_backend
        self.__shuffle = shuffle
        self.__shots = shots
        self.__epochs = epochs
        self.__optimizer = optimizer
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__hl_preproc = hidden_layers_preproc
        self.__hl_postproc = hidden_layers_postproc
        self.__reps_qlayer = repetitions_qlayer
        self.__shift = shift
        self.__weight_initialization = weight_initialization
        self.__w2w = w2w
        return

    def create_classifier(self, position_matrix : np.matrix, labels: list, similarity_matrix : np.matrix) -> np.matrix:
        n_samples = len(labels)
        n_classes = len(list(set(labels)))
        dimensions = position_matrix.shape[1]

        if n_classes > 2:
            raise Exception("Multi-class support for "+ str(type(self)) +" not implemented, yet.")

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
        model = DressedQNN(dimensions, n_classes, self.__hl_preproc,
                           self.__hl_postproc, self.__reps_qlayer,
                           self.__weight_initialization, self.__w2w, self.__shift,
                           backend, self.__shots)

        optimizer = self.instanciate_optimizer(self.__optimizer, model.parameters(), self.__learning_rate)

        loss_func = nn.NLLLoss()

        loss_list = []

        model.train()
        for epoch in range(self.__epochs):
            model.update_epoch(epoch)

            if self.__shuffle:
                shuff = list(range(len(position_matrix)))
                random.shuffle(shuff)
                position_matrix, labels = position_matrix[shuff], labels[shuff]

            total_loss = []
            for batch_idx, (data, target) in enumerate(zip(chunk(position_matrix, self.__batch_size), chunk(labels, self.__batch_size))):
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

        def prediction_fun(data):
            data_tensor = torch.tensor(data).float()
            data_res = model(data_tensor)

            lbls = [int(torch.argmax(res)) for res in data_res]
            return np.array(lbls)

        return prediction_fun, []

    def instanciate_optimizer(self, name, parameters, learningrate):
        return globals()[name](parameters, lr=learningrate)

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

    def get_hl_preproc(self):
        return self.__hl_preproc

    def set_hl_preproc(self, hl_preproc):
        self.__hl_preproc = hl_preproc

    def get_hl_postproc(self):
        return self.__hl_postproc

    def set_hl_postproc(self, hl_postproc):
        self.__hl_postproc = hl_postproc

    def get_reps_qlayer(self):
        return self.__reps_qlayer

    def set_reps_qlayer(self, reps_qlayer):
        self.__reps_qlayer = reps_qlayer

    def get_optimizer(self):
        return self.__optimizer

    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer

    def get_shift(self):
        return self.__shift

    def set_shift(self, shift):
        self.__shift = shift

    def get_weight_init(self):
        return self.__weight_initialization

    def set_weight_init(self, weight_init):
        self.__weight_initialization = weight_init

    def get_w2w(self):
        return self.__w2w

    def set_w2w(self, w2w):
        self.__w2w = w2w

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

        parameter_optimizer = self.get_optimizer()
        description_optimizer = "Optimizer : {'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'Rprop', 'RMSprop', 'LBFGS'}, (default='Adam')\n"\
                                    +"The classical optimizer to use."
        params.append(("optimizer", "Optimizer", description_optimizer, parameter_optimizer, "select", ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'SGD', 'Rprop', 'RMSprop', 'LBFGS']))

        parameter_learningrate = self.get_learningrate()
        description_learningrate = "Learning rate parameter"
        params.append(("learningrate", "Learning rate", description_learningrate, parameter_learningrate, "number", 0, 1e-3))

        parameter_epochs = self.get_epochs()
        description_epochs = "Epochs : int, (default=10)\n"\
                            +"Number of learning epochs."
        params.append(("epochs", "Epochs" , description_epochs, parameter_epochs, "number", 1, 1))

        parameter_batchsize = self.get_batchsize()
        description_batchsize = "Batch size : int, (default=10)\n"\
                            +"Batch size for training; It's highly recommendable to choose a batch size that is a factor of the number of samples"
        params.append(("batchsize", "Batchsize" , description_batchsize, parameter_batchsize, "number", 1, 1))

        parameter_shuffle = self.get_shuffle()
        description_shuffle = "shuffle: bool, (default=True)\n If True: randomly shuffle data before training"
        params.append(("shuffle", "Shuffle" , description_shuffle, parameter_shuffle, "checkbox"))

        parameter_preproc_layers = str(self.get_hl_preproc())
        description_preproc_layers = "Preprocessing: Classical hidden layer sizes : tuple, (default=(4,))\n"\
                            + "The ith element represents the number of neurons in the ith hidden layer."\
                            + "This is for the classical network that preprocesses the data before passing it to the quantum layer."
        params.append(("pre_hiddenlayersizes", "Preprocessing: Classical hidden layer sizes" , description_preproc_layers, parameter_preproc_layers, "text", "", ""))

        parameter_postproc_layers = str(self.get_hl_postproc())
        description_postproc_layers = "Postprocessing: Classical hidden layer sizes : tuple, (default=(4,))\n"\
                            + "The ith element represents the number of neurons in the ith hidden layer."\
                            + "This is for the classical network that postprocesses the data coming out of the quantum layer."
        params.append(("post_hiddenlayersizes", "Postprocessing: Classical hidden layer sizes" , description_postproc_layers, parameter_postproc_layers, "text", "", ""))

        parameter_reps_qlayer = self.get_reps_qlayer()
        description_reps_qlayer = "Quantum Layer: Number of repetitions: int, (default=4)\n"\
                                + "Repetitions in the trainable circuit of the quantum layer."\
                                + "This directly influences the depth of the overall circuit."
        params.append(("reps_qlayer", "Quantum Layer: Number of repetitions", description_reps_qlayer, parameter_reps_qlayer, "number", 1, 1))

        parameter_shift = round(self.get_shift(), 3)
        description_shift = "Shift: int, (default=π/4)\n"\
                            "Shift to apply for determination of gradients via parameter shift rule."
        params.append(("shift", "Quantum Layer: Shift for gradient determination", description_shift, parameter_shift, "number", 1e-3, 1e-3))

        parameter_weight_init = self.get_weight_init()
        description_weight_init = "Weight initialization: {'standard_normal', 'uniform', 'zero'}, (default='uniform')\n"\
                                    +"Distribution for (random) initialization of weights."
        params.append(("weight_init", "Quantum Layer: Weight initialization strategy", description_weight_init, parameter_weight_init, "select", ['standard_normal', 'uniform', 'zero']))

        parameter_w2w = self.get_w2w()
        description_w2w = "w2w: weights to wiggle: int, (default=0)\n"\
                                + "The number of weights in the quantum circuit to update in one optimization step. 0 means all."
        params.append(("w2w", "Quantum Layer: Weights to wiggle", description_w2w, parameter_w2w, "number", 0, 1))

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
            if param[0] == "optimizer":
                self.set_optimizer(param[3])
            if param[0] == "learningrate":
                self.set_learningrate(param[3])
            if param[0] == "batchsize":
                self.set_batchsize(param[3])
            if param[0] == "shuffle":
                self.set_shuffle(param[3])
            if param[0] == "pre_hiddenlayersizes":
                self.set_hl_preproc((make_tuple(param[3])))
            if param[0] == "post_hiddenlayersizes":
                self.set_hl_postproc((make_tuple(param[3])))
            if param[0] == "reps_qlayer":
                self.set_reps_qlayer(param[3])
            if param[0] == "shift":
                self.set_shift(param[3])
            if param[0] == "weight_init":
                self.set_weight_init(param[3])
            if param[0] == "w2w":
                self.set_w2w(param[3])

    def d2_plot(self, last_sequenz: List[int] , costumes: List[Costume]) -> None:
        pass


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
