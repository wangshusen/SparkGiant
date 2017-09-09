package distopt.logisticl1l2.Giant

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
 * Solve a logistic regression problem using prox-GIANT. 
 * Objective function is the mean of 
 *      f_j (w) = log (1 + exp(-z_j)) + gamma1*||w||_1 + 0.5*gamma2*||w||_2^2, 
 *      where z_j = <x_j, w>.
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])]) extends distopt.logisticl1l2.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = false
    val nInv: Double = 1.0 / n.toDouble
    var g: Array[Double] = new Array[Double](this.d)
            
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    println("There are " + rdd.count.toString + " partition.")

    /**
     * Train a logistic regression model.
     *
     * @param gamma1 the regularization parameter of the L1 norm
     * @param gamma2 the regularization parameter of the squared L2 norm
     * @param maxIterOuter max number of outer iterations
     * @param maxIterInner max number of inner iterations
     * @param lipchitz inversely proportional to learning rate
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma1: Double, gamma2: Double, maxIterOuter: Int, maxIterInner: Int, lipchitz: Double): (Array[Double], Array[Double], Array[Double]) = {
        this.gamma1 = gamma1
        this.gamma2 = gamma2
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma1, gamma2);
                                                 exe.setParams(lipchitz, maxIterInner);
                                                 exe})
                                    .persist()
        rddTrain.count
        println("Prox-GIANT: gamma1 = " + gamma1.toString + ", gamma2 = " + gamma2.toString + ", maxIterInner = " + maxIterInner.toString + ", lipchitz=" + lipchitz.toString)
        val t0: Double = System.nanoTime()
        
        // initialization
        for (j <- 0 until this.d) this.w(j) = 0.0
        
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIterOuter)
        val objValArray: Array[Double] = new Array[Double](maxIterOuter)
        val timeArray: Array[Double] = new Array[Double](maxIterOuter)
        
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIterOuter) {
            timeArray(t) = (t1 - t0) * 1.0E-9
            this.update(rddTrain)
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
            
            if (!this.isMute) println("Iteration " + t.toString + ":\t objective value is " + this.objVal.toString + ",\t time: " + timeArray(t).toString)
            
            if (timeArray(t) > this.timeOut || this.objVal > 1E10) {
                return (trainErrorArray.slice(0, t+1), 
                        objValArray.slice(0, t+1), 
                        timeArray.slice(0, t+1))
            }
            
        }
        
        val nnz: Double = this.w.filter(a => math.abs(a) > 1E-10).size.toDouble
        println("Sparsity of w is " + (nnz / this.d.toDouble).toString)
        
        (trainErrorArray, objValArray, timeArray)
    }
    
    /**
     * Take one approximate proximal Newton step.
     *
     * Update:
     *  1. this.w
     *  2. this.trainError
     *  3. this.objVal
     *
     * @param rddTrain RDD of executors
     * @return
     */
    def update(rddTrain: RDD[Executor]): Unit ={
        // update gradient, objective value, and training error
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        this.updateGradient(rddTrain, wBc)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        
        // compute the averaged Newton direction
        val p: Array[Double] = rddTrain.map(exe => exe.proxNewton(wBc.value, gBc.value))
                                    .reduce((a,b) => (a,b).zipped.map(_ + _)) 
                                    .map(_ * this.nInv)
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(p)
        
        // backtracking line search
        val eta: Double = this.linesearch(rddTrain, wBc, pBc)
        
        // take approximate Newton step
        for (j <- 0 until this.d) w(j) -= eta * p(j)
    }
    
    /**
     * Update the gradient, objective value, and training error.
     * Note that this.g is the gradient of logis(w) + 0.5*gamma2*||w||_2^2.
     */
    def updateGradient(rddTrain: RDD[Executor], wBc: Broadcast[Array[Double]]): Unit ={
        // compute full gradient
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ * this.nInv)
        for (j <- 0 until this.d) this.g(j) += gamma2 * this.w(j)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv
    }

    /** 
     * Search for a step size that leads to sufficient decrease.
     */
    def linesearch(rddTrain: RDD[Executor], wBc: Broadcast[Array[Double]], pBc: Broadcast[Array[Double]]): Double ={
        val objVals: Array[Double] = rddTrain
                        .map(_.objFunVal(wBc.value, pBc.value))
                        .reduce((a,b) => (a,b).zipped.map(_ + _))
                        .map(_ * this.nInv)

        //var pg: Double = 0.0
        //val p: Array[Double] = pBc.value
        //for (j <- 0 until this.d) pg += p(j) * this.g(j)
        //val eta: Double = this.lineSearch(objVals, -0.01 * pg)
        val eta: Double = this.lineSearch(objVals, 0)
        println("Eta = " + eta.toString)
        eta
    }
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.logisticl1l2.Common.Executor(arr) {
    val sDouble: Double = this.s.toDouble
            
    var lipchitz: Double = 1.0
    var maxIter: Int = 100
    def setParams(l0: Double, m0: Int): Unit = {
        this.lipchitz = l0
        this.maxIter = m0
    }
      
    /**
     * Minimize the model w.r.t. p:
     *      g'*p + 0.5*p'*H*p + gamma1*||w+p||_1, 
     *      where H = A * A' / s + gamma2*Identity.
     *
     * @param wArray the current iteration
     * @param gArray the gradient of l(w) = logis(w) + 0.5*gamma2*||w||_2^2 at w
     * @return -p
     */
    def proxNewton(wArray: Array[Double], gArray: Array[Double]): Array[Double] = {
        val g: DenseVector[Double] = new DenseVector(gArray)
        val w: DenseVector[Double] = new DenseVector(wArray)
        
        // compute Hessian matrix
        val zexp: Array[Double] = (this.x.t * w).toArray.map((a: Double) => math.exp(a))
        val ddiag: Array[Double] = zexp.map((r: Double) => (math.sqrt(r) / (1.0 + r)))
        val a: DenseMatrix[Double] = new DenseMatrix[Double](this.d, this.s)
        for (j <- 0 until this.s) {
            a(::, j) := ddiag(j) * this.x(::, j)
        }
        
        // solve the sub-problem by accelerated proximal SDCA
        val y: DenseVector[Double] = DenseVector.zeros[Double](this.s)
        val v: DenseVector[Double] = a * (a.t * w) * (1.0 / this.s.toDouble) + this.gamma2 * w - g
        val u: DenseVector[Double] = distopt.utils.SdcaL1L2.acceSdcaQuadratic(a, y, v / this.gamma2, this.gamma2, this.gamma1 / this.gamma2, this.maxIter, w)
        
        (w - u).toArray.map((b: Double) => b * this.s)
    }
    
}



