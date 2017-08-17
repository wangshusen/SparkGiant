package distopt.quadratic.Lbfgs

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
 * Solve a ridge regression problem using conjugate gradient (CG) method. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isModelAvg is true if model averaging is used to initialize w
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isModelAvg: Boolean = false)
        extends distopt.quadratic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = false
            
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
     * Train a ridge regression model using GIANT with the local problems solved by fixed number of CG steps.
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
        println("count = " + rddTrain.count.toString)
        
        // initialize w by model averaging
        if (this.isModelAvg) {
            this.w = rddTrain.map(_.solve())
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ / this.n.toDouble)
            println("Driver: model averaging is done!")
        }
        else {
            for (j <- 0 until this.d) this.w(j) = 0
        }
        
        // initialize the buffers and gradient
        var wBc: Broadcast[Array[Double]] = this.initialize(rddTrain)
        
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIter)
        val objValArray: Array[Double] = new Array[Double](maxIter)
        val timeArray: Array[Double] = new Array[Double](maxIter)
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIter) {
            timeArray(t) = (t1 - t0) * 1.0E-9
            wBc = this.update(wBc, rddTrain)
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
            
            if (this.gNorm < this.gNormTol) {
                return (trainErrorArray.slice(0, t+1), 
                        objValArray.slice(0, t+1), 
                        timeArray.slice(0, t+1))
            }
            if (!this.isMute) println("Iteration " + t.toString + ":\t objective value is " + this.objVal.toString + ",\t time: " + timeArray(t).toString)
        }
        
        (trainErrorArray, objValArray, timeArray)
    }
            
    /**
     * Initialize the gradient and buffers.
     *
     * @param rddTrain RDD of executors
     * @return
     */ 
    def initialize(rddTrain: RDD[Executor]): Broadcast[Array[Double]] ={
        // empty the buffers
        this.sBuffer = new Queue[DenseVector[Double]]
        this.yBuffer = new Queue[DenseVector[Double]]
        this.syBuffer = new Queue[Double]
        
        // evaluate gradient and objectives
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        this.updateGradient(wBc, rddTrain)
        
        wBc
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
    def update(wBcOld: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Broadcast[Array[Double]] ={
        // store old gradient
        this.gold := this.gnew
        
        // compute ascending direction by the two-loop iteration
        this.pnew := this.twoLoop()
        
        // search for a step size that leads to sufficient decrease
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(this.pnew.toArray)
        val eta: Double = this.wolfeLineSearch(wBcOld, pBc, rddTrain)
        //println("Eta = " + eta.toString)
        
        // update w
        for (j <- 0 until this.d) this.w(j) -= eta * this.pnew(j)
        
        // compute new gradient
        val wBcNew: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        this.updateGradient(wBcNew, rddTrain)
        
        // update buffers
        val st: DenseVector[Double] = (-eta) * this.pnew
        val yt: DenseVector[Double] = this.gnew - this.gold
        this.sBuffer += st
        this.yBuffer += yt
        var sy: Double = 0.0
        for (j <- 0 until this.d) sy += st(j) * yt(j)
        this.syBuffer += sy
        if (this.sBuffer.size > this.numHistory){
            this.sBuffer.dequeue
            this.yBuffer.dequeue
            this.syBuffer.dequeue
        }
        
        wBcNew
    }
        
    /** 
     * Evaluate gradient, objective value, and training error.
     *
     * @param wBc broadcast of this.w
     * @param rddTrain RDD of executors
     * @return
     */    
    def updateGradient(wBc: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Unit ={
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        val g: Array[Double] = tmp._1.map(_ * this.nInv)
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv
        for (j <- 0 until this.d) this.gnew(j) = g(j)
        
        this.gNorm = g.map(a => a*a).sum
        println("Driver: squared norm of gradient is " + this.gNorm.toString)
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
               
        def pDot(q: DenseVector[Double]): Double = {
            var pq: Double = 0.0
            for (j <- 0 until this.d) pq += p(j) * q(j)
            pq
        }
        
        val a: Array[Double] = new Array[Double](k)
        for (i <- 0 until k) {
            var j = k - 1 - i
            val aj: Double = pDot(this.sBuffer(j)) / this.syBuffer(j)
            a(j) = aj
            p -= aj * this.yBuffer(j)
        }
        val alpha: Double = this.syBuffer(k-1) / (this.yBuffer(k-1).toArray.map(a=>a*a).sum)
        p *= alpha
        for (i <- 0 until k) {
            val bi: Double = pDot(this.yBuffer(i)) / this.syBuffer(i)
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
     * @return step size
     */
    def wolfeLineSearch(wBc: Broadcast[Array[Double]], pBc: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Double = {
        val tmp: (Array[Double], Array[Double]) = rddTrain
                        .map(_.getAllObjAndGrad(wBc.value, pBc.value))
                        .reduce((a,b) => ((a._1,b._1).zipped.map(_ + _), (a._2,b._2).zipped.map(_ + _)))
        val objArray: Array[Double] = tmp._1.map(_ * this.nInv)
        val pgArray: Array[Double] = tmp._2.map(_ * this.nInv)
        
        //val pg: Double = this.pnew.t * this.gnew
        var pg: Double = 0.0
        for (j <- 0 until this.d) pg += this.pnew(j) * this.gnew(j)
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
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.quadratic.Common.Executor(arr) {
    val pgArray: Array[Double] = new Array[Double](this.numStepSizes)
            
    /**
     * Compute: 
     * (1) the local objective functions,
     * (2) the inner products of p and the gradients of the local objective functions,
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
            var res: DenseVector[Double] = this.x.t * wTmp - this.y
            // objective
            val trainError: Double = res.toArray.map(a => a*a).sum
            val wNorm: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = (trainError + sgamma * wNorm) * 0.5
            // gradient
            val g: DenseVector[Double] = this.x * res + sgamma * wTmp
            // the inner product <p, g>
            var pg: Double = 0.0
            for (j <- 0 until this.d) pg += p(j) * g(j)
            this.pgArray(idx) = pg
        }
        
        (this.objValArray, this.pgArray)
    }
}


