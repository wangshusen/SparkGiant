package distopt.quadratic.GiantExact

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

import distopt.quadratic.Executor

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
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ / this.n.toDouble)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        
        val gNorm: Double = g.map(a => a*a).sum
        println("Squared norm of gradient is " + gNorm.toString)
        
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
            val pg: Double = (this.p, this.g).zipped.map(_ * _).sum
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
    
    def predict(dataTest: RDD[(Double, Array[Double])]): Double = {
        val nTest: Long = dataTest.count
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        val error: Double = dataTest.map(pair => (pair._1, (pair._2, wBc.value).zipped.map(_ * _).sum))
                        .map(pair => (pair._1 - pair._2) * (pair._1 - pair._2))
                        .sum
        error / nTest.toDouble
    }

}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.quadratic.Executor.Executor(arr) {
    // initialization
    val svd.SVD(v, sig, _) = svd.reduced(this.x)
    var invH: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, d)
    
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


