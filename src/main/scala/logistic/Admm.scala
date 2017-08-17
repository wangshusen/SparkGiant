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
    val isMute: Boolean = true
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
        val rho: Double = 0.1 * gamma
        this.rho = rho
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);  
                                                 exe.setLocalMaxIter(q); 
                                                 exe.setLearningRate(learningRate);
                                                 exe})
                                    .persist()
        //println("Driver: executors are setup for training! gamma = " + gamma.toString + ", q = " + q.toString + ", learningRate = " + learningRate.toString)
        rddTrain.count
        val t0: Double = System.nanoTime()

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
            timeArray(t) = (t1 - t0) * 1.0E-9
            this.update(rddTrain)
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
            if (!this.isMute) println("Iteration " + t.toString + ":\t objective value is " + this.objVal.toString + ",\t time: " + timeArray(t).toString)
        }
        
        for (j <- 0 until this.d) this.w(j) = this.u(j)
        
        (trainErrorArray, objValArray, timeArray)
    }

    /**
     * Take one approximate ADMM descent step.
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
        
        // update w
        val tmp: Array[(Int, Array[Double], Double, Double)] = rddTrain.map(exe => exe.updateW(aBc.value, uBc.value, wBc.value, rhoBc.value)).collect
        this.wArrays = tmp.map(tuple => (tuple._1, tuple._2))
                        .sortWith(_._1 < _._1)
                        .map((pair: (Int, Array[Double])) => pair._2)
        
        // update the objective value and training error
        this.objVal = (tmp.map(tuple => tuple._3).sum) * this.nInv
        this.trainError = (tmp.map(tuple => tuple._4).sum) * this.nInv
        
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
        
        //if (this.rho < 1E4 * this.gamma) this.rho *= 1.1
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
     * As by-products, compute the objective value and training error.
     *
     * @param aArrays the dual variables
     * @param uArray slack variable
     * @param wArrays the primal variables
     * @param rho augmented Lagrangian parameter
     * @return the (index, primal variable, objective value, training error) tuple
     */    
    def updateW(aArrays: Array[Array[Double]], uArray: Array[Double], wArrays: Array[Array[Double]], rho: Double): (Int, Array[Double], Double, Double) = {
        val a: DenseVector[Double] = new DenseVector(aArrays(this.index))
        val wold: DenseVector[Double] = new DenseVector(wArrays(this.index))
        val u: DenseVector[Double] = new DenseVector(uArray)
        val diff: DenseVector[Double] = rho * (a - u)
        
        // compute objective value and training error
        val z: Array[Double] = (this.x.t * u).toArray
        val zexp: Array[Double] = z.map((a: Double) => math.exp(a))
        val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
        val uNorm: Double = uArray.map(a => a*a).sum
        val objVal: Double = loss + this.s * this.gamma * uNorm * 0.5
        val pred: Array[Double] = z.map((a: Double) => math.signum(a))
        val trainError: Double = z.filter(_ < 1E-30).length.toDouble
        
        // update W
        val wnew: Array[Double] = this.svrg(wold, diff, rho)
        (this.index, wnew, objVal, trainError)
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
