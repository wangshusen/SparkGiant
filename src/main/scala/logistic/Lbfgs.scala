package distopt.logistic.Lbfgs

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._
// scala data structure
import scala.collection.mutable.Queue

/**
 * Solve a logistic regression problem using L-BFGS 
 * Objective function is the mean of 
 * f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
 * where z_j = <x_j, w>.
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isModelAvg is true if model averaging is used to initialize w
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isModelAvg: Boolean = false)
        extends distopt.logistic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()

    val gold: DenseVector[Double] = DenseVector.zeros[Double](this.d)
    val gnew: DenseVector[Double] = DenseVector.zeros[Double](this.d)
    val pnew: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        
    var numHistory: Int = 10
    var sBuffer: Queue[DenseVector[Double]] = new Queue[DenseVector[Double]]
    var yBuffer: Queue[DenseVector[Double]] = new Queue[DenseVector[Double]]
    var syBuffer: Queue[Double] = new Queue[Double]
            
    /**
     * Train a logistic regression model using L-BFGS.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param numHistory number of saved history vectors
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, numHistory0: Int = 10): (Array[Double], Array[Double], Array[Double]) = {
        val t0: Double = System.nanoTime()
        
        this.numHistory = numHistory0
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);
                                                 exe})
                                    .persist()
        
        // initialize w by model averaging
        if (isModelAvg) {
            val q: Int = 100
            val learningRate: Double = 1.0
            this.w = rddTrain.map(_.solve(learningRate, q))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ * this.nInv)
            println("Driver: model averaging is done!")
            
        }
        else {
            for (j <- 0 until this.d) this.w(j) = 0.0
        }
        
        // initialize the buffers
        this.initialize(rddTrain)
        
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

    /**
     * Take one L-BFGS step.
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
        // store old gradient
        this.gold := this.gnew
        
        // compute ascending direction by two-loop iteration
        this.pnew := this.twoLoop()
        for (j <- 0 until this.d) this.p(j) = this.pnew(j)
        
        // search for a step size that leads to sufficient decrease
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(this.p)
        val eta: Double = this.wolfeLineSearch(wBc, pBc, rddTrain)
        //println("Eta = " + eta.toString)
        
        // update w
        for (j <- 0 until this.d) this.w(j) -= eta * this.p(j)
        
        // compute new gradient
        this.updateGradient(wBc, rddTrain)
        for (j <- 0 until this.d) this.gnew(j) = this.g(j)
        
        // update buffers
        val st: DenseVector[Double] = (-eta) * this.pnew
        val yt: DenseVector[Double] = this.gnew - this.gold
        this.sBuffer += st
        this.yBuffer += yt
        this.syBuffer += st.t * yt
        if (this.sBuffer.size > this.numHistory){
            this.sBuffer.dequeue
            this.yBuffer.dequeue
            this.syBuffer.dequeue
        }
    }
        
    /** 
     * Evaluate gradient, objective value, and training error.
     *
     * @param wBc broadcast of this.w
     * @param rddTrain RDD of executors
     * @return
     */    
    def updateGradient(wBc: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Unit ={
        // compute full gradient
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ * this.nInv)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv
    }
            
    /**
     * Initialize the gradient and buffers.
     *
     * @param rddTrain RDD of executors
     * @return
     */ 
    def initialize(rddTrain: RDD[Executor]): Unit ={
        // empty the buffers
        this.sBuffer = new Queue[DenseVector[Double]]
        this.yBuffer = new Queue[DenseVector[Double]]
        this.syBuffer = new Queue[Double]
        
        // evaluate gradient and objectives
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        this.updateGradient(wBc, rddTrain)
        for (j <- 0 until this.d) this.gnew(j) = this.g(j)
    }
    
    /**
     * Two-loop update of L-BFGS.
     *
     * @return ascending direction
     */
    def twoLoop(): DenseVector[Double] = {
        val p: DenseVector[Double] = -this.gold
        val k: Int = sBuffer.size
        if (k == 0) return p
        
        val a: Array[Double] = new Array[Double](k)
        for (i <- 0 until k) {
            var j = k - 1 - i
            val aj: Double = (p.t * this.sBuffer(j)) / this.syBuffer(j)
            a(j) = aj
            p -= aj * this.yBuffer(j)
        }
        val alpha: Double = this.syBuffer(k-1) / (this.yBuffer(k-1).t * this.yBuffer(k-1))
        p *= alpha
        for (i <- 0 until k) {
            val bi: Double = (p.t * this.yBuffer(i)) / this.syBuffer(i)
            p += (a(i) - bi) * this.sBuffer(i)
        }
        -p
    }
    
    /**
     * Find a step size that satisfies Wolfe conditions.
     *
     * @param wBc broadcast of this.w
     * @param pBc broadcast of this.p
     * @param rddTrain RDD of executors
     * @return ascending direction
     */
    def wolfeLineSearch(wBc: Broadcast[Array[Double]], pBc: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Double = {
        val tmp: (Array[Double], Array[Double]) = rddTrain
                        .map(_.getAllObjAndGrad(wBc.value, pBc.value))
                        .reduce((a,b) => ((a._1,b._1).zipped.map(_ + _), (a._2,b._2).zipped.map(_ + _)))
        val objArray: Array[Double] = tmp._1.map(_ * this.nInv)
        val pgArray: Array[Double] = tmp._2.map(_ * this.nInv)
        
        val pg: Double = this.pnew.t * this.gnew
        val pg1: Double = pg * 0.1
        val pg2: Double = pg * 0.2
        var flag1: Boolean = false
        var flag2: Boolean = false
        var eta: Double = 1.0
        
        // Wolfe condition
        for (j <- 0 until this.numStepSizes) {
            eta = this.stepSizes(j)
            flag1 = (objArray(j) <= this.objVal - pg1 * eta)
            flag2 = (pgArray(j) >= pg2)
            if (flag1 && flag2) return eta
        }
        
        // if the search direction p does not lead to sufficient decrease,
        // then return the smallest step size in the candidate set.
        eta
    }
                 
    def pDot(q: DenseVector[Double]): Double = {
        var pq: Double = 0.0
        for (j <- 0 until this.d) pq += this.p(j) * q(j)
        pq
    }
}



/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.logistic.Common.Executor(arr) {
    val pgArray: Array[Double] = new Array[Double](this.numStepSizes)
            
    /**
     * Compute: 
     * (1) the sum of the objective function
     *      f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
     *      where z_j = <x_j, w-eta*p>,
     * (2) the inner products of p and the gradients of the above function,
     * for all eta in the candidate set.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def getAllObjAndGrad(wArray: Array[Double], pArray: Array[Double]): (Array[Double], Array[Double]) = {
        val w: DenseVector[Double] = new DenseVector(wArray)
        val p: DenseVector[Double] = new DenseVector(pArray)
        var wTmp: DenseVector[Double] = DenseVector.zeros[Double](d)
        val sgamma: Double = this.s * this.gamma
        
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            val zexp: Array[Double] = (this.x.t * wTmp).toArray.map((a: Double) => math.exp(a))
            // objective
            val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
            val wNorm: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = loss + sgamma * wNorm * 0.5
            // gradient
            val c: DenseVector[Double] = new DenseVector(zexp.map((a: Double) => -1.0 / (1.0 + a)))
            val g: DenseVector[Double] = this.x * c + sgamma * w
            // the inner product <p, g>
            this.pgArray(idx) = p.t * g
        }
        
        (this.objValArray, this.pgArray)
    }
        
}

