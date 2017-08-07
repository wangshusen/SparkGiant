package distopt.logistic.Admm

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a logistic regression problem using ADMM with the local problems inexactly solved by SVRG. 
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
    var rho: Double = 1.0
    var gamma: Double = 1.0
    val mInv: Double = 1.0 / this.m
    
    // initialize executors
    val rdd: RDD[Executor] = data.glom.zipWithIndex.map(pair => new Executor(pair._1, pair._2)).persist()
    //println("There are " + rdd.count.toString + " partition.")
    //println("Driver: executors are initialized using the input data!")
    
    // assert that the indices of executors are 0, ..., m-1 without gap
    val indexArray: Array[Int] = rdd.map(exe => exe.index).collect.sortWith(_ < _)
    for (i <- 0 until this.m.toInt) assert(i == indexArray(i))
    
    var u: Array[Double] = new Array[Double](d)
    val aArrays: Array[Array[Double]] = Array.ofDim[Double](this.m.toInt, this.d)
    var wArrays: Array[Array[Double]] = Array.ofDim[Double](this.m.toInt, this.d)

    /**
     * Train a logistic regression model using Dane with the local problems solved by fixed number of SVRG steps.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param q number of CG iterations
     * @param learningRate learning rate of SVRG
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, q: Int, learningRate: Double): (Array[Double], Array[Double], Array[Double]) = {
        this.gamma = gamma
        val rho: Double = gamma
        this.rho = rho
        
        val t0: Double = System.nanoTime()
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);  
                                                 exe.setLocalMaxIter(q); 
                                                 exe.setLearningRate(learningRate);
                                                 exe})
                                    .persist()
        //println("Driver: executors are setup for training! gamma = " + gamma.toString + ", q = " + q.toString + ", learningRate = " + learningRate.toString)
        
        
        // initialize a, w, u 
        aArrays.map(ai => ai.map(aij => 0.0))
        wArrays.map(wi => wi.map(wij => 0.0))
        if (isModelAvg) { // model averaging
            this.u = rddTrain.map(_.solve(learningRate, q))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ * this.nInv)
            println("Driver: model averaging is done!")
            
        }
        else {
            for (j <- 0 until this.d) this.u(j) = 0.0
        }
        
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
        
        for (j <- 0 until this.d) this.w(j) = this.u(j)
        
        (trainErrorArray, objValArray, timeArray.map(time => time*1.0E-9))
    }

    /* Take one approximate Dane descent step.
     *
     * Update:
     *  1. this.a, this.u, this.w
     *  2. this.trainError
     *  3. this.objVal
     *
     * @param rddTrain RDD of executors
     * @return
     */
    def update(rddTrain: RDD[Executor]): Unit ={
        // broadcast a, u, w, and rho
        val aBc: Broadcast[Array[Array[Double]]] = this.sc.broadcast(this.aArrays)
        val uBc: Broadcast[Array[Double]] = this.sc.broadcast(this.u)
        val wBc: Broadcast[Array[Array[Double]]] = this.sc.broadcast(this.wArrays)
        val rhoBc: Broadcast[Double] = this.sc.broadcast(this.rho)
        
        // update the training error and objective value
        var tmp: (Double, Double) = rddTrain.map(exe => exe.grad(uBc.value))
                                        .map(tuple => (tuple._2, tuple._3))
                                        .reduce((a, b) => (a._1+b._1, a._2+b._2))
        this.trainError = tmp._1 * this.nInv
        this.objVal = tmp._2 * this.nInv
        
        // update w
        this.wArrays = rddTrain.map(exe => exe.updateW(aBc.value, uBc.value, wBc.value, rhoBc.value))
                        .collect
                        .sortWith(_._1 < _._1)
                        .map((pair: (Int, Array[Double])) => pair._2)
        
        // update u (locally)
        val normalizer: Double = this.rho / (this.gamma + this.rho)
        val wAvg: Array[Double] = this.wArrays.reduce(this.arrayAdd).map(_ * mInv)
        val aAvg: Array[Double] = this.aArrays.reduce(this.arrayAdd).map(_ * mInv)
        for (j <- 0 until this.d) this.u(j) = (wAvg(j) + aAvg(j)) * normalizer
        
        // update a (locally)
        for (i <- 0 until this.m.toInt) {
            for (j <- 0 until this.d) {
                this.aArrays(i)(j) += this.wArrays(i)(j) - this.u(j)
            }
        }
        
        //if (this.rho < 100 * this.gamma) this.rho *= 1.1
    }
    
    def arrayAdd(arr1: Array[Double], arr2: Array[Double]): Array[Double] = {
        val arr: Array[Double] = arr1.clone
        for (j <- 0 until this.d) arr(j) += arr2(j)
        arr
    }
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])], idx: Long) extends 
        distopt.logistic.Common.Executor(arr) {
    // constants
    val index: Int = idx.toInt
            
    // tuning parameters
    var learningRate: Double = 1.0
    def setLearningRate(learningRate0: Double): Unit = {
        this.learningRate = learningRate0
    }
         
    /**
     * Locally update w_i.
     *
     * @param aArrays the dual variables
     * @param uArray slack variable
     * @param wArrays the primal variables
     * @param rho augmented Lagrangian parameter
     * @return the (index, primal variable) pair
     */    
    def updateW(aArrays: Array[Array[Double]], uArray: Array[Double], wArrays: Array[Array[Double]], rho: Double): (Int, Array[Double]) = {
        val a: DenseVector[Double] = new DenseVector(aArrays(this.index))
        val wold: DenseVector[Double] = new DenseVector(wArrays(this.index))
        val diff: DenseVector[Double] = rho * (a - wold)
        
        val wnew: Array[Double] = this.svrg(wold, diff, rho)
        (this.index, wnew)
    }
    
            
    /**
     * Update w_i by SVRG.
     *
     * @param wold the current solution as initialization
     * @param diff rho * (a_i - u)
     * @param rho augmented Lagrangian parameter
     * @return w the new primal variable
     */ 
    def svrg(wold: DenseVector[Double], diff: DenseVector[Double], rho: Double): Array[Double] = {
        // parameters that can be tuned
        val sizeBatch: Int = 128
        val invSizeBatch: Double = -1.0 / sizeBatch
        val numInnerLoop: Int = math.floor(this.s / sizeBatch).toInt
        val invS: Double = -1.0 / this.s.toDouble
        
        // Shuffle the columns of X
        val randIndex: List[Int] = scala.util.Random.shuffle((0 until this.s).toList)
        val xShuffle: DenseMatrix[Double] = DenseMatrix.zeros[Double](this.d, this.s)
        for (j <- 0 until this.s) {
            xShuffle(::, j) := this.x(::, randIndex(j))
        }
        
        // buffers
        val w: DenseVector[Double] = wold.copy
        val wtilde: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val z: DenseVector[Double] = DenseVector.zeros[Double](this.s)
        val c: DenseVector[Double] = DenseVector.zeros[Double](this.s)
        val zRand: DenseVector[Double] = DenseVector.zeros[Double](sizeBatch)
        val cRand: DenseVector[Double] = DenseVector.zeros[Double](sizeBatch)
        val gRand: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val gExact: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val xsample: DenseMatrix[Double] = DenseMatrix.zeros[Double](this.d, sizeBatch)
        
        
        for (innerIter <- 0 until this.q) {
            wtilde := w
            
            // exact local gradient
            z := xShuffle.t * wtilde
            for (i <- 0 until this.s) c(i) = invS / (1.0 + math.exp(z(i)))
            gExact := xShuffle * c
            
            for (j <- 0 until numInnerLoop) {
                xsample := xShuffle(::, j*sizeBatch until (j+1)*sizeBatch)
                gRand := gExact
                
                // stochastic gradient at w
                zRand := xsample.t * w
                for (i <- 0 until sizeBatch) cRand(i) = invSizeBatch / (1.0 + math.exp(zRand(i)))
                gRand += xsample * cRand + diff + rho * w
                
                // stochastic gradient at wtilde
                zRand := xsample.t * wtilde
                for (i <- 0 until sizeBatch) cRand(i) = invSizeBatch / (1.0 + math.exp(zRand(i)))
                gRand -= xsample * cRand
                
                // update w
                w -= this.learningRate * gRand
            }
        }
        
        //z := xShuffle.t * w
        //for (i <- 0 until this.s) c(i) = invS / (1.0 + math.exp(z(i)))
        //gExact := xShuffle * c + diff + rho * w
        //val gNorm: Double = gExact.toArray.map(a => a*a).sum
        //println("gNorm = " + gNorm.toString)
        
        w.toArray
    }
            
           
        
}
