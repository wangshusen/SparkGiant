package distopt.ridge.GiantExact

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
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
    
    // setup executors
    var rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    
    // for line search
    val stepSizes: Array[Double] = (0 until 10).toArray.map(1.0 / math.pow(4, _))
    val numStepSizes: Int = stepSizes.length
    if (isSearch) {
        val stepSizesBc = sc.broadcast(stepSizes)
        this.rdd = this.rdd.map(exe => {exe.setStepSizes(stepSizesBc.value); exe}).persist()
    }
    
    /**
     * Train a ridge regression model using GIANT with the local problems exactly solved.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int): (Array[Double], Array[Double], Array[Double]) = {
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIter)
        val objValArray: Array[Double] = new Array[Double](maxIter)
        val timeArray: Array[Double] = new Array[Double](maxIter)
        
        val t0: Double = System.nanoTime()
        
        // initialize the executors
        this.rdd = this.rdd.map(exe => {exe.setGamma(gamma); exe.solve; exe}).persist()
        
        // initialize w by model averaging
        this.w := this.rdd.map(exe => exe.w).reduce((a, b) => a+b) * (1.0 / this.m)
        val wBc = this.sc.broadcast(this.w)
        this.rdd = this.rdd.map(exe => {exe.setW(wBc.value); exe}).persist()
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIter) {
            timeArray(t) = t1 - t0
            this.update
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
        }
        
        (trainErrorArray, objValArray, timeArray.map(time => time*1.0E-9))
    }

    // take one approximate Newton step
    def update(): Unit ={
        // compute full gradient
        var tmp = this.rdd.map(exe => exe.grad).reduce((a, b) => (a._1+b._1, a._2+b._2, a._3+b._3))
        this.gFull := tmp._1 * (1.0 / this.n)
        this.trainError = tmp._2 * (1.0 / this.n)
        this.objVal = tmp._3 * (1.0 / this.n)
        var gBc = this.sc.broadcast(gFull)

        // compute the averaged Newton direction
        pFull := this.rdd.map(exe => exe.newton(gBc.value)).reduce((a, b) => a+b) * (1.0 / this.n)
        var pBc = this.sc.broadcast(pFull)
        this.rdd = this.rdd.map(exe => {exe.setP(pBc.value); exe}).persist()
        
        // take Newton step
        if (isSearch) { // search for a step size that leads to sufficient decrease
            val pg: Double = -0.1 * sum(pFull :* gFull)
            var eta: Double = this.lineSearch(pg)
            this.rdd = this.rdd.map(exe => {exe.updateW(eta); exe}).persist()
            this.w -= eta * pFull
        }
        else { // use step size 1.0
            this.rdd = this.rdd.map(exe => {exe.updateW(); exe}).persist()
            this.w -= pFull
        }
    }
    
    /** 
     * Search for the best step size eta
     *
     * @param pg = -0.1 * <p, g>
     * @return eta the best step size
     */
    def lineSearch(pg: Double): Double = {
        var eta: Double = 0.0
        
        // get the objective values f(w - eta*p) for all eta in the candidate list
        val objValArray: Array[Double] = this.rdd
                            .map(exe => exe.objFunVal)
                            .reduce((a,b) => (a zip b).map(pair => pair._1+pair._2))
                            .map(_ / this.n.toDouble)
        
        // backtracking line search (Armijo rule)
        for (j <- 0 until numStepSizes) {
            eta = stepSizes(j)
            var objValNew = objValArray(j)
            // sufficient decrease in the objective value
            if (objValNew < this.objVal + pg * eta) { 
                println("objval is " + objValNew.toString)
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
    var gamma: Double = 0.0
    var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
    var p: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
    var invH: DenseMatrix[Double] = DenseMatrix.zeros[Double](1, 1)
    
    // for line search
    var stepSizes: Array[Double] = Array(1.0)
    var numStepSizes: Int = 1
    var objValArray: Array[Double] = Array[Double](1)

    println("Local sample size is " + s.toString)

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
    }

    def setP(p0: DenseMatrix[Double]): Unit = {
        this.p := p0
    }

    def setW(w0: DenseMatrix[Double]): Unit = {
        this.w := w0
    }

    def updateW(eta: Double = 1.0): Unit = {
        this.w -= eta * this.p
    }
    
    // for line search
    def setStepSizes(ss: Array[Double]): Unit = {
        this.stepSizes = ss
        this.numStepSizes = ss.length
    }

    /**
     * Compute the local objective function value
     *      0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2
     * for all w = this.w - eta*p
     */
    def objFunVal(): Array[Double] = {
        val objValArray: Array[Double] = new Array[Double](this.numStepSizes)
        var wTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var res: DenseMatrix[Double] = DenseMatrix.zeros[Double](s, 1)
        
        for (j <- 0 until this.numStepSizes) {
            wTmp := this.w - this.stepSizes(j) * this.p
            res := this.x.t * wTmp - this.y
            objValArray(j) = (sum(res :* res) + this.s * this.gamma * sum(wTmp :* wTmp)) / 2.0
        }
        
        objValArray
    }

    /**
     * 1. Compute the local gradient  
     *      g = X' * (X * w - y) + s * gamma * w
     *
     * 2. As a by-product, compute the local training error
     *      ||X w - y||_2^2
     *
     * 3. As a by-product, compute the local objective function value
     *      0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2
     */
    def grad(): (DenseMatrix[Double], Double, Double) = {
        val res = this.x.t * this.w - this.y
        val g = this.x * res + (this.s * this.gamma) * this.w
        // the training error and objective value are by-products
        val trainError = sum(res :* res)
        val wNorm = sum(this.w :* this.w)
        val objVal = (trainError + this.s * this.gamma * wNorm) / 2
        (g, trainError, objVal)
    }

    /**
     * 1. Compute the inverse Hessian matrix (X'*X/s + gamma*I)^{-1}
     *
     * 2. Optimize the ridge regression problem 
     * 0.5/s*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
     * using the local data.
     */
    def solve(): Unit = {
        val svd.SVD(v, sig, _) = svd.reduced(this.x)
        val sig2: DenseVector[Double] = (sig :* sig) * (1.0/this.s) + this.gamma
        this.invH = (v(*, ::) :/ sig2) * v.t
        this.w := (this.invH * (x * y)) * (1.0 / s)
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
