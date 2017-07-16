package distopt.quadratic.GiantExact

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a ridge regression problem using GIANT with the local problems exactly solved. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isSearch is true if line search is used to determine the step size; otherwise use 1.0 as step size
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isSearch: Boolean = false)
        extends distopt.quadratic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    
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
                                    .map(exe => {exe.setGamma(gamma);  
                                                 exe.invertHessian; 
                                                 exe})
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

    /* Take one approximate Newton step.
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
        var eta = 1.0
        if (isSearch) { 
            // get the objective values f(w - eta*p) for all eta in the candidate list
            val objVals: Array[Double] = rddTrain
                            .map(_.objFunVal(wBc.value, pBc.value))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ / this.n.toDouble)
            
            val pg: Double = (this.p, this.g).zipped.map(_ * _).sum
            eta = this.lineSearch(objVals, -0.1 * pg)
        }
        
        // take approximate Newton step
        this.w = (this.w, this.p).zipped.map((a, b) => a - eta*b)
    }
    
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.quadratic.Common.Executor(arr) {
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
        val w: DenseMatrix[Double] = this.invH * this.xy
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


