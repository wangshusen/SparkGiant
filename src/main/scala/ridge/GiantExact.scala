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
    var w: Array[Double] = new Array[Double](d)
    var g: Array[Double] = new Array[Double](d)
    var p: Array[Double] = new Array[Double](d)
    var trainError: Double = 0.0
    var objVal: Double = 0.0
    
    // for line search
    var eta: Double = 0.0
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
        println("count = " + rddTrain.count.toString)
        println("Driver: executors are setup for training! gamma = " + gamma.toString)
        
        // initialize w by model averaging
        this.w = rddTrain.map(_.solve())
                        .reduce((a,b) => (a,b).zipped.map(_ + _))
                        .map(_ / this.n.toDouble)
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

    // Take one approximate Newton step.
    def update(rddTrain: RDD[Executor]): Unit ={
        // broadcast w
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        
        // compute full gradient
        var tmp = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ / this.n.toDouble)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        
        // update the training error and objective value
        this.trainError = tmp._2 * (1.0 / this.n)
        this.objVal = tmp._3 * (1.0 / this.n)

        // compute the averaged Newton direction
        this.p = rddTrain.map(exe => exe.newton(gBc.value.toArray))
                        .reduce((a,b) => (a,b).zipped.map(_ + _)) 
                        .map(_ / this.n.toDouble)
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(this.p)
        
        // search for a step size that leads to sufficient decrease
        if (isSearch) { 
            val pg: Double = (this.p, this.g).zipped.map(_ * _).reduce(_ + _)
            this.eta = this.lineSearch(rddTrain, -0.1 * pg, wBc, pBc)
        }
        else {
            this.eta = 1.0
        }
        
        // take approximate Newton step
        this.w = (this.w, this.p).zipped.map((a, b) => a - eta*b)
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
    def lineSearch(rddTrain: RDD[Executor], pg: Double, wBc: Broadcast[Array[Double]], pBc: Broadcast[Array[Double]]): Double = {
        var eta: Double = 0.0
        
        // get the objective values f(w - eta*p) for all eta in the candidate list
        val objVals: Array[Double] = rddTrain
                            .map(_.objFunVal(wBc.value, pBc.value))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
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
    var invH: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, d)
    
    println("Executor: initialized!")

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
    }

    /**
     * Compute the local objective function values
     *      0.5*||X (w - eta*p) - y||_2^2 + 0.5*s*gamma*||(w - eta*p)||_2^2
     * for all eta in the candidate set.
     * This function is for line search.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def objFunVal(wArray: Array[Double], pArray: Array[Double]): Array[Double] = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        val p: DenseMatrix[Double] = new DenseMatrix(this.d, 1, pArray)
        var wTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var res: DenseMatrix[Double] = DenseMatrix.zeros[Double](s, 1)
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            res := this.x.t * wTmp - this.y
            var trainError: Double = res.toArray.map(a => a*a).reduce(_ + _)
            var wNorm: Double = wTmp.toArray.map(a => a*a).reduce(_ + _)
            this.objValArray(idx) = (trainError + this.s * this.gamma * wNorm) / 2.0
        }
        
        this.objValArray
    }

    /**
     * Compute the local gradient.
     * As by-products, also compute the training error and objective value.
     *
     * @param w the current solution
     * @return g = X' * (X * w - y) + s * gamma * w , the local gradient
     * @return trainError = ||X w - y||_2^2 , the local training error
     * @return objVal = 0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2 , the local objective function value
     */
    def grad(wArray: Array[Double]): (Array[Double], Double, Double) = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        // gradient
        var res: DenseMatrix[Double] = this.x.t * w
        res := res - this.y
        var g: DenseMatrix[Double] = this.x * res 
        g := g + (this.s * this.gamma) * w
        // training error
        val trainError: Double = res.toArray.map(a => a*a).reduce(_ + _)
        // objective function value
        val wNorm: Double = w.toArray.map(a => a*a).reduce(_ + _)
        val objVal: Double = (trainError + this.s * this.gamma * wNorm) / 2
        (g.toArray, trainError, objVal)
    }

    
    /**
     * Compute the inverse Hessian matrix (X'*X/s + gamma*I)^{-1}
     * using the local data.
     */
    def invertHessian(): Unit = {
        var sig2: DenseVector[Double] = (this.sig :* this.sig) * (1.0/this.s) + this.gamma
        sig2 := 1.0 / sig2
        for (j <- 0 until d) {
            this.invH(::, j) := this.v(::, j) :* sig2(j)
        }
        this.invH := this.invH * this.v.t
    }
    
    /**
     * Optimize the ridge regression problem 
     * 0.5/s*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
     * using the local data.
     *
     * @return w solution to the local problem
     */
    def solve(): Array[Double] = {
        val w: DenseMatrix[Double] = this.invH * (this.x * this.y)
        w.toArray
    }

    /**
     * Compute the local Newton direction
     *
     * @param gArray the full gradient
     * @return the local Newton direction scaled by s
     */
    def newton(gArray: Array[Double]): Array[Double] = {
        val g: DenseMatrix[Double] = new DenseMatrix(this.d, 1, gArray)
        val p: DenseMatrix[Double] = (this.invH * g) * this.s.toDouble
        p.toArray
    }
}
