package distopt.logistic.Dane

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a logistic regression problem using Dane with the local problems inexactly solved by SVRG. 
 * Objective function is the mean of 
 * f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
 * where z_j = <x_j, w>.
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isSearch is true if line search is used to determine the step size; otherwise use 1.0 as step size
 * @param isModelAvg is true if model averaging is used to initialize w
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isSearch: Boolean = false, isModelAvg: Boolean = false)
        extends distopt.logistic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = true
            
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    //println("There are " + rdd.count.toString + " partition.")
    //println("Driver: executors are initialized using the input data!")

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
        
        // initialize w by model averaging
        if (isModelAvg) {
            this.w = rddTrain.map(_.solve(learningRate, q))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ * this.nInv)
            println("Driver: model averaging is done!")
            
        }
        else {
            for (j <- 0 until this.d) this.w(j) = 0.0
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
            
            if (this.gNorm < this.gNormTol || timeArray(t) > this.timeOut) {
                return (trainErrorArray.slice(0, t+1), 
                        objValArray.slice(0, t+1), 
                        timeArray.slice(0, t+1))
            }
        }
        
        (trainErrorArray, objValArray, timeArray)
    }

    /* Take one approximate Dane descent step.
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
        this.g = tmp._1.map(_ * this.nInv)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        
        this.gNorm = g.map(a => a*a).sum
        //println("Driver: squared norm of gradient is " + this.gNorm.toString)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv

        // compute the averaged descending direction
        this.p = rddTrain.map(exe => exe.svrg(wBc.value, gBc.value))
                        .reduce((a,b) => (a,b).zipped.map(_ + _)) 
                        .map(_ / this.m.toDouble)
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(this.p)
        
        // search for a step size that leads to sufficient decrease
        var eta: Double = 1.0
        if (isSearch) { 
            // get the objective values f(w - eta*p) for all eta in the candidate list
            val objVals: Array[Double] = rddTrain
                            .map(_.objFunVal(wBc.value, pBc.value))
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ * this.nInv)
            
            var pg: Double = 0.0
            for (j <- 0 until this.d) pg += this.p(j) * this.g(j)
            eta = this.lineSearch(objVals, -0.01 * pg)
            //println("Eta = " + eta.toString)
        } 
        
        // take approximate Newton step
        for (j <- 0 until this.d) w(j) -= eta * this.p(j)
    }
    
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.logistic.Common.Executor(arr) {
    var learningRate: Double = 1.0
    def setLearningRate(learningRate0: Double){
        this.learningRate = learningRate0
    }   
            
    /**
     * Compute descent direction using the local data.
     *
     * @param wArray the current solution
     * @param gArray the full gradient
     * @return p the descent direction
     */ 
    def svrg(wArray: Array[Double], gArray: Array[Double]): Array[Double] = {
        // parameters that can be tuned
        val sizeBatch: Int = 128
        val invSizeBatch: Double = -1.0 / sizeBatch
        val numInnerLoop: Int = math.floor(this.s / sizeBatch).toInt
        val invS: Double = -1.0 / this.s.toDouble
        
        // compute LocalGradient minus FullGradient
        val wold: DenseVector[Double] = new DenseVector(wArray)
        val z: DenseVector[Double] = this.x.t * wold
        val zexp: Array[Double] = z.toArray.map((a: Double) => math.exp(a))
        val c: DenseVector[Double] = new DenseVector(zexp.map((a: Double) => -1.0 / (1.0 + a)))
        val gDiff: DenseVector[Double] = (this.x * c) * this.sInv + this.gamma * wold // local gradient
        for (i <- 0 until this.d) gDiff(i) -= gArray(i)
        
        // Shuffle the columns of X
        val randIndex: List[Int] = scala.util.Random.shuffle((0 until this.s).toList)
        val xShuffle: DenseMatrix[Double] = DenseMatrix.zeros[Double](this.d, this.s)
        for (j <- 0 until this.s) {
            xShuffle(::, j) := this.x(::, randIndex(j))
        }
        
        // buffers
        val w: DenseVector[Double] = wold.copy
        val wtilde: DenseVector[Double] = DenseVector.zeros[Double](this.d)
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
                gRand += this.gamma * w - gDiff + xsample * cRand
                
                // stochastic gradient at wtilde
                zRand := xsample.t * wtilde
                for (i <- 0 until sizeBatch) cRand(i) = invSizeBatch / (1.0 + math.exp(zRand(i)))
                gRand -= xsample * cRand
                
                // update w
                w -= this.learningRate * gRand
            }
        }
        
        (wold - w).toArray
    }
            
           
        
}
