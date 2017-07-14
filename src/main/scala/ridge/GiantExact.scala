package distopt.ridge.GiantExact

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math


/**
 * Solve a ridge regression problem using GIANT with the local problems exactly solved. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isSearch is true if line search is used to determine the step size; otherwise use 1.0 as step size
 */
class Driver(sc: SparkContext, var data: RDD[(Double, Array[Double])], isSearch: Boolean = false) {
    // constants
    val n: Long = data.count
    val d: Int = data.take(1)(0)._2.size
    val m: Long = data.getNumPartitions
    
    // variables
    var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
    var gFull: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
    var pFull: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
    var trainError: Double = 0.0
    var objVal: Double = 0.0
    
    // for line search
    val numStepSizes: Int = 10
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(4, _))
    val stepSizesBc = sc.broadcast(stepSizes)
    
    // initialize executors
    val t0: Double = System.nanoTime()
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    println("Driver: executors are initialized using the input data!")
    
    /**
     * Train a ridge regression model using GIANT with the local problems exactly solved.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int): (Array[Double], Array[Double], Array[Double]) = {
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma); exe.invertHessian; exe})
                                    .persist()
        println("Driver: executors are setup for training! gamma = " + gamma.toString)
        
        // initialize w by model averaging
        this.w := rddTrain.map(_.solve())
                        .reduce((a, b) => a+b)
        this.w *= (1.0 / this.m)
        println("Driver: model averaging is done!")
        
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIter)
        val objValArray: Array[Double] = new Array[Double](maxIter)
        val timeArray: Array[Double] = new Array[Double](maxIter)
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIter) {
            timeArray(t) = t1 - t0
            this.update(rddTrain)
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
        }
        
        (trainErrorArray, objValArray, timeArray.map(time => time*1.0E-9))
    }

    // take one approximate Newton step
    def update(rddTrain: RDD[Executor]): Unit ={
        // broadcast w
        val wBc: Broadcast[DenseMatrix[Double]] = this.sc.broadcast(this.w)
        
        // compute full gradient
        var tmp = rddTrain.map(exe => exe.grad(wBc.value)).reduce((a, b) => (a._1+b._1, a._2+b._2, a._3+b._3))
        this.gFull := tmp._1 * (1.0 / this.n)
        this.trainError = tmp._2 * (1.0 / this.n)
        this.objVal = tmp._3 * (1.0 / this.n)
        
        // broadcast g
        val gBc: Broadcast[DenseMatrix[Double]] = this.sc.broadcast(gFull)

        // compute the averaged Newton direction
        pFull := rddTrain.map(exe => exe.newton(gBc.value)).reduce((a, b) => a+b) * (1.0 / this.n)
        
        // broadcast p
        val pBc: Broadcast[DenseMatrix[Double]] = this.sc.broadcast(pFull)
        
        // take approximate Newton step
        if (isSearch) { // search for a step size that leads to sufficient decrease
            val pg: Double = -0.1 * sum(pFull :* gFull)
            var eta: Double = this.lineSearch(rddTrain, pg, wBc, pBc)
            this.w -= eta * pFull
        }
        else { // use step size 1.0
            this.w -= pFull
        }
    }
    
    /** 
     * Search for the best step size eta
     *
     * @param rddTrain RDD of executors
     * @param pg = -0.1 * <p, g>
     * @param wBc the broadcast of w
     * @param pBc the broadcast of p
     * @return eta the best step size
     */
    def lineSearch(rddTrain: RDD[Executor], pg: Double, wBc: Broadcast[DenseMatrix[Double]], pBc: Broadcast[DenseMatrix[Double]]): Double = {
        var eta: Double = 0.0
        
        // get the objective values f(w - eta*p) for all eta in the candidate list
        val objVals: Array[Double] = rddTrain
                            .map(_.objFunVal(wBc.value, pBc.value))
                            .reduce((a,b) => (a zip b).map(pair => pair._1+pair._2))
                            .map(_ / this.n.toDouble)
        
        // backtracking line search (Armijo rule)
        for (j <- 0 until this.numStepSizes) {
            eta = this.stepSizes(j)
            var objValNew = objVals(j)
            // sufficient decrease in the objective value
            if (objValNew < this.objVal + pg * eta) { 
                return eta
            }
        }
        
        // if the search direction p does not lead to sufficient decrease,
        // then return the smallest step size in the candidate set.
        eta
    }

}

/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(var arr: Array[(Double, Array[Double])]) {
    // get data
    val s: Int = arr.size
    val d: Int = arr(0)._2.size
    val y: DenseMatrix[Double] = new DenseMatrix(s, 1, arr.map(pair => pair._1))
    val x: DenseMatrix[Double] = new DenseMatrix(d, s, arr.map(pair => pair._2).flatten)

    // initialization
    val svd.SVD(v, sig, _) = svd.reduced(this.x)
    
    // for line search
    // make sure stepSizes is consistent with the one defined in the driver
    val numStepSizes: Int = 10
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(4, _))
    val objValArray: Array[Double] = new Array[Double](this.numStepSizes)

    // specific to training
    var gamma: Double = 0.0
    var invH: DenseMatrix[Double] = DenseMatrix.zeros[Double](1, 1)
    
    println("Executor: initialized!")

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
    }

    /**
     * Compute the local objective function value
     *      0.5*||X (w - eta*p) - y||_2^2 + 0.5*s*gamma*||(w - eta*p)||_2^2
     * for all eta in the candidate set.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def objFunVal(w: DenseMatrix[Double], p: DenseMatrix[Double]): Array[Double] = {
        var wTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var res: DenseMatrix[Double] = DenseMatrix.zeros[Double](s, 1)
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            res := this.x.t * wTmp - this.y
            this.objValArray(idx) = (sum(res :* res) + this.s * this.gamma * sum(wTmp :* wTmp)) / 2.0
        }
        
        this.objValArray
    }

    /**
     * @param w the current solution
     * @return g = X' * (X * w - y) + s * gamma * w , the local gradient
     * @return trainError = ||X w - y||_2^2 , the local training error
     * @return objVal = 0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2 , the local objective function value
     */
    def grad(w: DenseMatrix[Double]): (DenseMatrix[Double], Double, Double) = {
        val res = this.x.t * w - this.y
        val g = this.x * res + (this.s * this.gamma) * w
        // the training error and objective value are by-products
        val trainError = sum(res :* res)
        val wNorm = sum(w :* w)
        val objVal = (trainError + this.s * this.gamma * wNorm) / 2
        (g, trainError, objVal)
    }

    
    /**
     * Compute the inverse Hessian matrix (X'*X/s + gamma*I)^{-1}
     * using the local data.
     */
    def invertHessian(): Unit = {
        var sig2: DenseVector[Double] = (this.sig :* this.sig) * (1.0/this.s) + this.gamma
        sig2 = sig2.map(1.0 / _)
        this.invH = this.v.copy
        this.invH(*, ::) :*= sig2
        this.invH := this.invH * this.v.t
    }
    
    /**
     * Optimize the ridge regression problem 
     * 0.5/s*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
     * using the local data.
     *
     * @return w solution to the local problem
     */
    def solve(): DenseMatrix[Double] = {
        (this.invH * (this.x * this.y)) * (1.0 / s)
    }

    /**
     * Compute the local Newton direction
     *
     * @param gFull the full gradient
     * @return the local Newton direction scaled by s
     */
    def newton(gFull: DenseMatrix[Double]): DenseMatrix[Double] = {
        (this.invH * gFull) * this.s.toDouble
    }

}
