from qiskit import *
from qiskit.visualization import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import random
from backend.logger import Logger

class BaseQuantumKMeans():
    """
    This is the base implementation for quantum k means
    implementations on qiskit using the algorithms from
    https://arxiv.org/abs/1909.12183.

    The StatePreparationQuantumKMeans
    is a different quantum Kmeans algorithm that has
    been developed by my own.
    """

    def __init__(self):
        ## Plotting variables
        self.runSafe = 0
        self.circuitAmountSafe = 0
        self.currentCircuitSafe = 0
        self.residualSafe = 100.0

        ## projection axis
        self.xAxis = (1.0, 0.0)
        self.yAxis = (0.0, 1.0)
        self.xyAxis = (-1/2 * sqrt(2), 1/2 * sqrt(2))

        self.baseVector = self.xAxis

        # Parameters
        self.k = 2
        self.maxQubits = 2
        self.shotsEach = 100
        self.maxRuns = 10
        self.relativeResidualAmount = 5

        return

    def ExecuteNegativeRotation(self, dataRaw, k, maxQubits, shotsEach, maxRuns, relativeResidualNumber, backend, plotData, plotCircuit, whichCircuit = "negrot"):
        """
        Executes the quantum k means cluster algorithm on the given
        quantum backend and the specified circuit.
        The dataRaw needs to be 2D cartesian coordinates.
        ShotsEach describes how often one circuit will be executed.
        We return a list with a mapping from data indizes to centeroid indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        """
        centroidsRaw = self.GenerateRandomData(k)

        # plot the initial data before mapping to unit sphare
        if plotData:
            self.PlotRawData([dataRaw, centroidsRaw], ["Raw data", "Raw centroids"], "Initial raw data", "InitialRaw", whichCircuit)

        # map to unit sphere
        centroids = self.Normalize(self.Standardize(centroidsRaw))
        data = self.Normalize(self.Standardize(dataRaw))

        # plot the initial data after mapping to unit sphare
        if plotData:
            self.PlotData([data, centroids], ["Preprocessed data", "Preprocessed centroids"], "Preprocessed data", "InitialPreprocessed", whichCircuit)

        # data angles don't need to be updated in every iteration,
        # they are fixed
        dataAngles = self.CalculateAngles(data)

        oldCentroidMapping = np.zeros(len(data))
        newCentroidMapping = np.zeros(len(data))
        counter = 1
        notConverged = True
        globalAmountExecutedCircuits = 0
        while notConverged and counter < maxRuns:
            self.UpdateConsole(run = counter)
            #print("run no " + str(counter))
            centroidAngles = self.CalculateAngles(centroids)
            oldCentroidMapping = newCentroidMapping
            shouldPlotCircuit = 1 if plotCircuit and counter == 1 else False
            newCentroidMapping = None
            amountExecutedCircuits = 0
            if whichCircuit == "negrot":
                (newCentroidMapping, amountExecutedCircuits) = self.ApplyNegativeRotationCircuit(centroidAngles, dataAngles, maxQubits, shotsEach, backend, shouldPlotCircuit)
            elif whichCircuit == "inter":
                (newCentroidMapping, amountExecutedCircuits) = self.ApplyDestructiveInterferenceCircuit(centroidAngles, dataAngles, maxQubits, shotsEach, backend, shouldPlotCircuit)
            elif whichCircuit == "custom":
                (newCentroidMapping, amountExecutedCircuits) = self.ApplyStatePreparationQuantumKMeansCircuit(centroids, data, maxQubits, shotsEach, backend, shouldPlotCircuit)
            globalAmountExecutedCircuits += amountExecutedCircuits
            notConverged = not self.CheckConvergency(oldCentroidMapping, newCentroidMapping, relativeResidualNumber)
            if notConverged:
                centroids = self.CalculateCentroids(newCentroidMapping, centroids, data, k)
                centroids = self.Normalize(centroids)
                counter += 1
            else:
                notConverged = False
        
        if counter >= maxRuns:
            loggingString = "Stopped with " \
                + str(counter) \
                + " iterations (" \
                + str(self.residualSafe) + " % residual" \
                + ") and executed " \
                + str(globalAmountExecutedCircuits) \
                + " circuits of " \
                + whichCircuit \
                + " in total with " \
                + str(globalAmountExecutedCircuits * shotsEach) \
                + " shots in total using " \
                + str(maxQubits) \
                + " qubits in parallel on " \
                + backend.name() \
                + " backend."
            Logger.normal(loggingString)
        else:
            loggingString = "Converged with " \
                + str(counter) \
                + " iterations (" \
                + str(self.residualSafe) + " % residual" \
                + ") and executed " \
                + str(globalAmountExecutedCircuits) \
                + " circuits of " \
                + whichCircuit \
                + " in total with " \
                + str(globalAmountExecutedCircuits * shotsEach) \
                + " shots in total using " \
                + str(maxQubits) \
                + " qubits in parallel on " \
                + backend.name() \
                + " backend."
            Logger.normal(loggingString)

        if plotData:
            # MAKE DATA PREPARATION FOR PLOTS
            dataTexts = []
            clusters = dict()
            clustersRaw = dict()

            for i in range(0, len(newCentroidMapping)):
                clusterNumber = int(newCentroidMapping[i])
                if clusterNumber not in clusters:
                    clusters[clusterNumber] = []
                    clustersRaw[clusterNumber] = []
                
                clusters[clusterNumber].append(data[i])
                clustersRaw[clusterNumber].append(dataRaw[i])

            # add missing clusters that have no elements
            for i in range(0, k):
                if i not in clusters:
                    clusters[i] = []

            clustersPlot = []
            clustersRawPlot = []

            for i in range(0, k):
                clustersPlot.append([])
                clustersRawPlot.append([])
                for j in range(0, len(clusters[i])):
                    clustersPlot[i].append(clusters[i][j])
                    clustersRawPlot[i].append(clustersRaw[i][j])

            for i in range(0, k):
                dataTexts.append("Cluster" + str(i))

            self.PlotData(clustersPlot, dataTexts, "Preprocessed clusters after " + str(counter) + " iterations (" + str(self.residualSafe) + " % residual)", "ClusterPreprocessed", whichCircuit)
            self.PlotRawData(clustersRawPlot, dataTexts, "Raw clusters after " + str(counter) + " iterations (" + str(self.residualSafe) + " % residual)", "ClusterRaw", whichCircuit)

        return newCentroidMapping

    def GenerateRandomData(self, amount):
        """
        Generate amount many random 2D data pints.
        """
        data = np.array([None] * amount) # store tuple as coordinates

        # Create random float numbers per coordinate
        for i in range(0, amount):
            data[i] = (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

        return data

    def Standardize(self, data):
        """
        Standardize all the points, i.e. they have zero mean and unit variance.
        Note that a copy of the data points will be created.
        """
        dataX = np.zeros(len(data))
        dataY = np.zeros(len(data))
        preprocessedData = np.array([None] * len(data))

        # create x and y coordinate arrays
        for i in range(0, len(data)):
            dataX[i] = data[i][0]
            dataY[i] = data[i][1]

        # make zero mean and unit variance, i.e. standardize
        tempDataX = (dataX - dataX.mean(axis=0)) / dataX.std(axis=0)
        tempDataY = (dataY - dataY.mean(axis=0)) / dataY.std(axis=0)

        # create tuples and normalize
        for i in range(0, len(data)):
            preprocessedData[i] = (tempDataX[i], tempDataY[i])

        return preprocessedData

    def Normalize(self, data):
        """
        Normalize the data, i.e. every entry of data has length 1.
        Note, that a copy of the data will be done.
        """
        preprocessedData = np.array([None] * len(data))

        # create tuples and normalize
        for i in range(0, len(data)):
            norm = sqrt(pow(data[i][0], 2) + pow(data[i][1], 2))
            preprocessedData[i] = (data[i][0] / norm, data[i][1] / norm)

        return preprocessedData

    def CalculateCentroids(self, centroidMapping, oldCentroids, data, k):
        """
        Calculates the new cartesian positions of the
        given centroids in the centroid mapping.
        """
        centroids = np.array([None] * k) # store tuple as coordinates per centroid

        for i in range(0, k):
            sumX = 0.0
            sumY = 0.0
            amount = 0
            for j in range(0, len(centroidMapping)):
                if centroidMapping[j] == i:
                    sumX += data[j][0]
                    sumY += data[j][1]
                    amount += 1

            # if no points assigned to centroid, take old coordinates
            if amount == 0:
                averagedX = oldCentroids[i][0]
                averagedY = oldCentroids[i][1]
            else:
                averagedX = sumX / amount
                averagedY = sumY / amount
            norm = sqrt(pow(averagedX, 2) + pow(averagedY, 2))
            centroids[i] = (averagedX / norm, averagedY / norm)
        return centroids

    def CheckConvergency(self, oldCentroidMapping, newCentroidMapping, relativeResidualNumber):
        """
        Check wether two centroid mappings are different and how different they are
        i.e. this is the convergency condition. Return true if we are converged.
        The relativeResidualNumber is the percentage of how many points are
        allowed to have a different label in the new iteration but still accept it 
        as converged.

        E.g.: residual = 0.05, 100 data points in total

        =>  if from one iteration to the other less than 100 * 0.05 = 5 points change their
            label, we still accept it as converged
        """
        countOfDifferentLabels = 0
        amountOfDataPoints = len(newCentroidMapping)
        for i in range(0, len(oldCentroidMapping)):
            if oldCentroidMapping[i] != newCentroidMapping[i]:
                countOfDifferentLabels += 1

        self.UpdateConsole(residual = round(countOfDifferentLabels / amountOfDataPoints * 100, 2))

        if countOfDifferentLabels < ceil(amountOfDataPoints * relativeResidualNumber):
            return True
        else:
            return False

    def CalculateAngles(self, cartesianPoints):
        """
        Calculates the angle between the 2D vetors and the base vector.
        The cartesian points are given in a tuple format (x, y).
        """
        angles = np.zeros(len(cartesianPoints))
        for i in range(0, len(angles)):
            # formula: alpha = acos( 1/(|a||b|) * a • b )
            # here: |a| = |b| = 1
            angles[i] = acos(self.baseVector[0] * cartesianPoints[i][0] + self.baseVector[1] * cartesianPoints[i][1])
        return angles

    def ApplyNegativeRotationCircuit(self, centroidAngles, dataAngles, maxQubits, shotsEach, backend, plot):
        """
        Create and apply the negative rotation circuit.
        Note, that the lenght of centeroids is the same as k!

        We return a list with a mapping from test angle indizes to centeroid angle indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        ...

        We need for each data angle len(centroidAngles) qubits.
        We do this in a chained fession, i.e. we take the first
        data angle and use len(centroidAngles) qubits.
        If we still have free qubits, we take the next data angle
        and do the same. If we reach the maxQubits limit, we execute the
        circuit and safe the result. If we are not able to test all centeroids
        in one run, we just use the next run for the remaining centeroids.
        """

        # this is also the amount of qubits that are needed in total.
        # note that this is not necessarily in parallel, also sequential
        # is possible here.
        globalWorkAmount = len(centroidAngles) * len(dataAngles)
        # we store the results as [(t1,c1), (t1,c2), ..., (t1,cn), (t2,c1), ..., (tm,cn)]
        # while each (ti,cj) stands for one floating point number, i.e. P|0> for one qubit
        result = np.zeros(globalWorkAmount)

        # create tuples of parameters corresponding for each qubit,
        # i.e. create [t1,c1, t1,c2, ..., t1,cn, t2,c1, ..., tm,cn]
        # now with ti = DataAngle_i and cj = CenteroidAngle_j
        parameters = []
        for i in range(0, len(dataAngles)):
            for j in range(0, len(centroidAngles)):
                parameters.append((dataAngles[i], centroidAngles[j]))

        queueNotEmpty = True
        index = 0 # this is the index to iterate over all parameter pairs in the queue (parameters list)
        circuitsIndex = 0
        amountExecutedCircuits = 0
        while queueNotEmpty:
            maxQubitsForCircuit = globalWorkAmount - index
            qubitsForCircuit = maxQubits if maxQubits < maxQubitsForCircuit else maxQubitsForCircuit
            qc = QuantumCircuit(qubitsForCircuit, qubitsForCircuit)

            for i in range(0, qubitsForCircuit):
                qc.ry( parameters[index][0], i) # testAngle rotation
                qc.ry(-parameters[index][1], i) # negative centeroidAngle rotation
                qc.measure(i, i)
                index += 1
                if index == globalWorkAmount:
                    queueNotEmpty = False
                    break
            if plot:
                qc.draw('mpl', filename='negrot_circuit' + str(circuitsIndex) + '.svg')
            circuitsIndex += 1
            self.UpdateConsole(circuitAmount = ceil(globalWorkAmount / maxQubits), currentCircuit = ceil(index / maxQubits))
            #print("Execute quantum circuit " + str(ceil(index / maxQubits)) + " / " + str(ceil(globalWorkAmount / maxQubits)), end="\r", flush=True)
            job = execute(qc, backend, shots=shotsEach)
            amountExecutedCircuits += 1
            histogram = job.result().get_counts()
            hits = self.CalculateP0Hits(histogram)
            for i in range(0, len(hits)):
                result[index - qubitsForCircuit + i] = hits[i]
        
        #print()

        centroidMapping = np.zeros(len(dataAngles))
        for i in range(0, len(dataAngles)):
            highestHitNumber = result[i * len(centroidAngles) + 0]
            highestHitCentroidIndex = 0
            for j in range(1, len(centroidAngles)):
                if result[i * len(centroidAngles) + j] > highestHitNumber:
                    highestHitCentroidIndex = j
            centroidMapping[i] = highestHitCentroidIndex
        
        return (centroidMapping, amountExecutedCircuits)

    def ApplyDestructiveInterferenceCircuit(self, centroidAngles, dataAngles, maxQubits, shotsEach, backend, plot):
        """
        Create and apply the distance calculation using
        destructive interference circuit.
        Note, that the lenght of centeroids is the same as k!

        We return a list with a mapping from test angle indizes to centeroid angle indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        ...

        We need for each centroid-data pair two qubits.
        We do this in a chained fession, i.e. we take the first
        data angle and use len(centroidAngles) qubits.
        If we still have free qubits, we take the next test angle
        and do the same. If we reach the maxQubits limit, we execute the
        circuit and safe the result. If we are not able to test all centeroids
        in one run, we just use the next run for the remaining centeroids.
        """

        # this is also the amount of qubits that are needed in total.
        # note that this is not necessarily in parallel, also sequential
        # is possible here. He need at least 2 qubits that run in parallel.
        globalWorkAmount = len(centroidAngles) * len(dataAngles)
        # we store the results as [(t1,c1), (t1,c2), ..., (t1,cn), (t2,c1), ..., (tm,cn)]
        # while each (ti,cj) stands for one floating point number, i.e. P|1> for the second qubit
        # i.e. P|11> + P|10>, the distance (not normed, proportional to distance)
        result = np.zeros(globalWorkAmount)

        # create tuples of parameters corresponding for each qubit,
        # i.e. create [t1,c1, t1,c2, ..., t1,cn, t2,c1, ..., tm,cn]
        # now with ti = DataAngle_i and cj = CenteroidAngle_j
        parameters = []
        for i in range(0, len(dataAngles)):
            for j in range(0, len(centroidAngles)):
                parameters.append((dataAngles[i], centroidAngles[j]))

        queueNotEmpty = True
        index = 0 # this is the index to iterate over all parameter pairs in the queue (parameters list)
        circuitsIndex = 0
        amountExecutedCircuits = 0
        while queueNotEmpty:
            maxQubitsForCircuit = (globalWorkAmount - index) * 2
            qubitsForCircuit = maxQubits if maxQubits < maxQubitsForCircuit else maxQubitsForCircuit
            if qubitsForCircuit % 2 != 0:
                qubitsForCircuit -= 1
            qc = QuantumCircuit(qubitsForCircuit, qubitsForCircuit)

            for i in range(0, qubitsForCircuit, 2):
                qc.h(i)
                qc.cx(i, i+1)
                relativeAngular = abs( parameters[index][0] - parameters[index][1] )
                qc.ry(-relativeAngular, i+1) # relative angular difference rotation
                qc.cx(i, i+1)
                qc.ry(relativeAngular, i+1) # relative angular difference rotation
                qc.h(i)
                qc.measure(i, i)
                qc.measure(i+1, i+1)
                index += 1
                if index == globalWorkAmount:
                    queueNotEmpty = False
                    break
            
            if plot:
                qc.draw('mpl', filename='inter_circuit' + str(circuitsIndex) + '.svg')
            circuitsIndex += 1
            self.UpdateConsole(circuitAmount = ceil(globalWorkAmount * 2 / maxQubits), currentCircuit = ceil(index * 2 / maxQubits))
            job = execute(qc, backend, shots=shotsEach)
            amountExecutedCircuits += 1
            histogram = job.result().get_counts()
            distances = self.CalculateP1Hits(histogram)
            for i in range(0, len(distances)):
                result[index - int(qubitsForCircuit / 2) + i] = distances[i]

        centroidMapping = np.zeros(len(dataAngles))
        for i in range(0, len(dataAngles)):
            lowestDistance = result[i * len(centroidAngles) + 0]
            lowestDistanceCentroidIndex = 0
            for j in range(1, len(centroidAngles)):
                if result[i * len(centroidAngles) + j] < lowestDistance:
                    lowestDistanceCentroidIndex = j
            centroidMapping[i] = lowestDistanceCentroidIndex
        
        return (centroidMapping, amountExecutedCircuits) 

    def ApplyStatePreparationQuantumKMeansCircuit(self, centroids, data, maxQubits, shotsEach, backend, plot):
        # this is also the amount of qubits that are needed in total.
        # note that this is not necessarily in parallel, also sequential
        # is possible here. He need at least 2 qubits that run in parallel.
        globalWorkAmount = len(centroids) * len(data)
        # we store the results as [(t1,c1), (t1,c2), ..., (t1,cn), (t2,c1), ..., (tm,cn)]
        # while each (ti,cj) stands for one floating point number, i.e. P|0> for the second, resp.
        # the first qubit (P|0> = P|00>)
        result = np.zeros(globalWorkAmount)

        # create tuples of parameters corresponding for each qubit,
        # i.e. create [t1,c1, t1,c2, ..., t1,cn, t2,c1, ..., tm,cn]
        # now with ti = DataAngle_i and cj = CenteroidAngle_j
        parameters = []
        for i in range(0, len(data)):
            for j in range(0, len(centroids)):
                theta_data = acos(data[i][0])
                theta_centroid = acos(centroids[j][0])
                parameters.append((theta_data, theta_centroid))

        queueNotEmpty = True
        index = 0 # this is the index to iterate over all parameter pairs in the queue (parameters list)
        circuitsIndex = 0
        amountExecutedCircuits = 0
        while queueNotEmpty:
            maxQubitsForCircuit = (globalWorkAmount - index) * 2
            qubitsForCircuit = maxQubits if maxQubits < maxQubitsForCircuit else maxQubitsForCircuit
            if qubitsForCircuit % 2 != 0:
                qubitsForCircuit -= 1
            qc = QuantumCircuit(qubitsForCircuit, qubitsForCircuit)

            for i in range(0, qubitsForCircuit, 2):
                qc.h(i)
                qc.cx(i, i+1)
                qc.ry(parameters[index][0], i)      # angle for data point
                qc.ry(parameters[index][1], i+1)    # angle for centroid
                qc.cx(i, i+1)
                qc.h(i)
                qc.measure(i, i)
                qc.measure(i+1, i+1)
                index += 1
                if index == globalWorkAmount:
                    queueNotEmpty = False
                    break
            
            if plot:
                qc.draw('mpl', filename='cus/circuit' + str(circuitsIndex) + '.svg')
            circuitsIndex += 1
            self.UpdateConsole(circuitAmount = ceil(globalWorkAmount * 2 / maxQubits), currentCircuit = ceil(index * 2 / maxQubits))
            job = execute(qc, backend, shots=shotsEach)
            amountExecutedCircuits += 1
            histogram = job.result().get_counts()
            distances = self.CalculateP0HitsOdd(histogram)
            for i in range(0, len(distances)):
                result[index - int(qubitsForCircuit / 2) + i] = distances[i]

        centroidMapping = np.zeros(len(data))
        for i in range(0, len(data)):
            minusDistance = result[i * len(centroids) + 0]
            lowestDistanceCentroidIndex = 0
            for j in range(1, len(centroids)):
                # now take the biggest value because
                # the probability is propotional to minus
                # the distance
                if result[i * len(centroids) + j] > minusDistance:
                    lowestDistanceCentroidIndex = j
            centroidMapping[i] = lowestDistanceCentroidIndex
        
        return (centroidMapping, amountExecutedCircuits) 


    def CalculateP0Hits(self, histogram):
        """
        Given a histogram from a circuit job result it calculates
        the hits for each qubit being measured in |0> state.
        """
        # we store for index i the hits of |0>,
        # i.e. hits[i] = #(qubit i measured in |0>)
        # the lenght is the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        length = len(list(histogram.keys())[0])
        hits = np.zeros(length)
        for basisState in histogram:
            for i in range(0, length):
                if basisState[length - i - 1] == "0":
                    hits[i] += histogram[basisState]
        return hits

    def MapHistogramToQubitHits(self, histogram):
        """
        """
        length = int(len(list(histogram.keys())[0]))
        # Create array and store tuple per qubit, i.e. (P|0>, P|1>)
        # the lenght is half the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        qubitHits = np.array([None] * length)
        for basisState in histogram:
            for i in range(0, length):
                if qubitHits[i] is None:
                    qubitHits[i] = (0.0, 0.0)

                if basisState[length - i - 1] == "0":
                    qubitHits[i] = (qubitHits[i][0] + histogram[basisState], qubitHits[i][1])
                else:
                    qubitHits[i] = (qubitHits[i][0], qubitHits[i][1] + histogram[basisState])
        
        return qubitHits

    def CalculateP1Hits(self, histogram):
        """
        Given a histogram from a circuit job result it calculates
        the hits for the odd qubits being measured in |1> state.
        """
        # we store for index i the hits of |1X>,
        # i.e. hits[i] = #(i-th odd qubit measured in |1>)
        # the lenght is half the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        length = int(len(list(histogram.keys())[0]) / 2)
        hits = np.zeros(length)

        qubitHits = self.MapHistogramToQubitHits(histogram)
        for i in range(0, int(len(qubitHits) / 2)):
            hits[i] = int(qubitHits[i * 2][1])

        # dont norm because we take the lowest value just as a label
        return hits

    def CalculateP0HitsOdd(self, histogram):
        """
        Given a histogram from a circuit job result it calculates
        the hits for the odd qubits being measured in |0> state.
        """
        # we store for index i the hits of |0X>,
        # i.e. hits[i] = #(i-th odd qubit measured in |0>)
        # the lenght is half the amount of qubits, that can be read out from the
        # string of any arbitrary (e.g. the 0th) bitstring
        length = int(len(list(histogram.keys())[0]) / 2)
        hits = np.zeros(length)

        qubitHits = self.MapHistogramToQubitHits(histogram)
        for i in range(0, int(len(qubitHits) / 2)):
            hits[i] = int(qubitHits[i * 2][0])

        # dont norm because we take the lowest value just as a label
        return hits

    def PlotRawData(self, dataLists, dataNames, title, fileName, filePrefix):
        """
        Plots raw data, i.e. no circle nor color as a 2D scatter plot
        Each entry in dataLists is a list of its own in the format [(x, y)].
        Each entryin dataNames is a string containing for the legend.
        Each entry in dataColors is a color code e.g. 'g'.
        """
        plt.clf()
        plt.cla()
        plt.figure(figsize=(7, 7), dpi=80)
        ax = plt.gca()
        ax.cla() # clear things for fresh plot
        ax.set_xlim(-1.5, +1.5)
        ax.set_ylim(-1.5, +1.5)
        ax.set_title(title)
        colors = self.GetColors(len(dataLists))
        for i in range(0, len(dataLists)):
            ax.scatter([dataPoints[0] for dataPoints in dataLists[i]], [dataPoints[1] for dataPoints in dataLists[i]], color=colors[i], label=dataNames[i])
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.savefig(filePrefix +"_" + fileName + ".svg")
        return

    def PlotData(self, dataLists, dataNames, title, fileName, filePrefix):
        """
        Plots as a 2D scatter plot with unit circle.
        Each entry in dataLists is a list of its own in the format [(x, y)].
        Each entryin dataNames is a string containing for the legend.
        Each entry in dataColors is a color code e.g. 'g'.
        """
        plt.clf()
        plt.cla()
        plt.figure(figsize=(7, 7), dpi=80)
        unitCirclePlot = plt.Circle((0, 0), 1.0, color='k', fill=False)
        ax = plt.gca()
        ax.cla() # clear things for fresh plot
        ax.set_xlim(-1.5, +1.5)
        ax.set_ylim(-1.5, +1.5)
        ax.set_title(title)
        ax.add_artist(unitCirclePlot)
        colors = self.GetColors(len(dataLists))
        for i in range(0, len(dataLists)):
            ax.scatter([dataPoints[0] for dataPoints in dataLists[i]], [dataPoints[1] for dataPoints in dataLists[i]], color=colors[i], label=dataNames[i])
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.savefig(filePrefix + "_" + fileName + ".svg")
        return

    def GetColors(self, k):
        """
        Return k colors in a list. We choose from 7 different colors.
        If k > 7 we choose colors more than once.
        """
        baseColors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        colors = []
        index = 1
        for i in range(0, k):
            if index % (len(baseColors) + 1) == 0:
                index = 1
            colors.append(baseColors[index - 1])
            index += 1
        return colors

    def UpdateConsole(self, run = None, circuitAmount = None, currentCircuit = None, residual = None):
        if run is None:
            run = self.runSafe
        else:
            self.runSafe = run
        if circuitAmount is None:
            circuitAmount = self.circuitAmountSafe
        else:
            self.circuitAmountSafe = circuitAmount
        if currentCircuit is None:
            currentCircuit = self.currentCircuitSafe
        else:
            self.currentCircuitSafe = currentCircuit
        if residual is None:
            residual = self.residualSafe
        else:
            self.residualSafe = residual
        
        printText  = "Iteration " + str(self.runSafe) + "     "
        printText += "Circuit " + str(self.currentCircuitSafe) + " / " + str(self.circuitAmountSafe) + "     "
        printText += "Current residual " + str(self.residualSafe) + " %          "
        print(printText, end="\r", flush=True)
        return

    # getter and setter methodes
    def get_number_of_clusters(self) -> int:
        return self.k

    def set_number_of_clusters(self, number_of_clusters : int = 2) -> None:
        self.k = number_of_clusters

    def get_max_qubits(self) -> int:
        return self.maxQubits

    def set_max_qubits(self, maxQubits : int = 2) -> None:
        self.maxQubits = maxQubits

    def get_shots_each(self) -> int:
        return self.shotsEach

    def set_shots_each(self, shotsEach : int = 100) -> None:
        self.shotsEach = shotsEach

    def get_max_runs(self) -> int:
        return self.maxRuns

    def set_max_runs(self, maxRuns : int = 10) -> None:
        self.maxRuns = maxRuns

    def get_relative_residual_amount(self) -> int:
        return self.relativeResidualAmount

    def set_relative_residual_amount(self, relativeResidualAmount : int = 5) -> None:
        self.relativeResidualAmount = relativeResidualAmount

    def get_backend(self):
        return self.backend

    def set_backend(self, backend) -> None:
        self.backend = backend

class NegativeRotationQuantumKMeans(BaseQuantumKMeans):
    """
    A class for applying the negative rotation k means
    algorithm on qiskit.
    """

    def __init__(self):
        super().__init__()
        return
    
    def Run(self, data, plotData = False, plotCircuit = False):
        """
        Runs the circuit and returns the cluster mapping, i.e.
        we return a list with a mapping from data indizes to cluster indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to cluster with index 2
        data vector with index 1 -> mapped to cluster 0
        data vector with index 2 -> mapped to cluster 1
        """
        clusterMapping = self.ExecuteNegativeRotation(data, self.k, self.maxQubits, self.shotsEach, self.maxRuns, self.relativeResidualAmount / 100.0, self.backend, plotData, plotCircuit, "negrot")
        return clusterMapping

class DestructiveInterferenceQuantumKMeans(BaseQuantumKMeans):
    """
    A class for applying the destructive interference k means
    algorithm on qiskit.
    """

    def __init__(self):
        super().__init__()
        return
    
    def Run(self, data, plotData = False, plotCircuit = False):
        """
        Runs the circuit and returns the cluster mapping, i.e.
        we return a list with a mapping from data indizes to cluster indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to cluster with index 2
        data vector with index 1 -> mapped to cluster 0
        data vector with index 2 -> mapped to cluster 1
        """
        clusterMapping = self.ExecuteNegativeRotation(data, self.k, self.maxQubits, self.shotsEach, self.maxRuns, self.relativeResidualAmount / 100.0, self.backend, plotData, plotCircuit, "inter")
        return clusterMapping

class StatePreparationQuantumKMeans(BaseQuantumKMeans):
    """
    A class for applying the destructive interference k means
    algorithm on qiskit.
    """

    def __init__(self):
        super().__init__()
        return
    
    def Run(self, data, plotData = False, plotCircuit = False):
        """
        Runs the circuit and returns the cluster mapping, i.e.
        we return a list with a mapping from data indizes to cluster indizes,
        i.e. if we return a list [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to cluster with index 2
        data vector with index 1 -> mapped to cluster 0
        data vector with index 2 -> mapped to cluster 1
        """
        clusterMapping = self.ExecuteNegativeRotation(data, self.k, self.maxQubits, self.shotsEach, self.maxRuns, self.relativeResidualAmount / 100.0, self.backend, plotData, plotCircuit, "custom")
        return clusterMapping

if __name__== "__main__":

    ### DEFINE QUANTUM BACKEND ###
    #provider = IBMQ.enable_account("")
    #backend = provider.get_backend("ibmq_athens")              #  5 qubits (VQ32)
    #backend = provider.get_backend("ibmq_5_yorktown")           #  5 qubits (VQ32)
    #backend = provider.get_backend("ibmq_16_melbourne")        # 15 qubits (VQ8)
    #backend = provider.get_backend("ibmq_qasm_simulator")      # 32 qubits
    backend = Aer.get_backend("qasm_simulator")
    maxQubits = 16 # This number defines the maximum qubits that hardware offers

    ### DEFINE GLOBAL PARAMETERS ###
    dimension = 2       # until now we can only use 2 dimensions, i.e. this is no parameter
    shotsEach = 8192
    k = 2               # The amount of clusters looking for, i.e. the number of centroids
    dataSize = 20       # The amount of random generated data points
    maxRuns = 10
    relativeResidualAmount = 5 # i.e. 5% of data points can be relabeled

    # Set the parameters
    negrot = DestructiveInterferenceQuantumKMeans()
    negrot.set_number_of_clusters(k)
    negrot.set_max_qubits(maxQubits)
    negrot.set_shots_each(shotsEach)
    negrot.set_max_runs(shotsEach)
    negrot.set_relative_residual_amount(relativeResidualAmount)
    negrot.set_backend(backend)

    ### DEFINE DATA ###
    data = negrot.GenerateRandomData(dataSize)

    ### EXECUTE THE DESIRED ALGORITHM ###
    centroidMapping = negrot.Run(data, True, False)
