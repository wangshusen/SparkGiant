package distopt.logistic.Agd

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a logistic regression problem using accelerated gradient descent 
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
            
    // initialize executors
    val rdd: RDD[distopt.logistic.Common.Executor] = data.glom.map(new distopt.logistic.Common.Executor(_)).persist()
    //println("There are " + rdd.count.toString + " partition.")
    //println("Driver: executors are initialized using the input data!")

    /**
     * Train a logistic regression model using GIANT with the local problems solved by fixed number of CG steps.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param learningRate learning rate
     * @param momentum (between 0 and 1)
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, learningRate: Double, momentum: Double): (Array[Double], Array[Double], Array[Double]) = {
        // setup the executors for training
        val rddTrain: RDD[distopt.logistic.Common.Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);
                                                 exe})
                                    .persist()
        //println("Driver: executors are setup for training! gamma = " + gamma.toString + ", maxIterOuter = " + maxIter.toString)
        rddTrain.count
        val t0: Double = System.nanoTime()
        
        
        // initialize w by model averaging
        if (isModelAvg) {
            val q: Int = 100
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
        
        for (j <- 0 until this.d) this.p(j) = 0.0
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIter) {
            timeArray(t) = (t1 - t0) * 1.0E-9
            this.update(rddTrain, learningRate, momentum)
            t1 = System.nanoTime()
            trainErrorArray(t) = this.trainError
            objValArray(t) = this.objVal
            
            if (!this.isMute) println("Iteration " + t.toString + ":\t objective value is " + this.objVal.toString + ",\t time: " + timeArray(t).toString)
        }
        
        (trainErrorArray, objValArray, timeArray)
    }

    /* Take one gradient descent step.
     *
     * Update:
     *  1. this.w
     *  2. this.p
     *  3. this.trainError
     *  4. this.objVal
     *
     * @param rddTrain RDD of executors
     * @return
     */
    def update(rddTrain: RDD[distopt.logistic.Common.Executor], learningRate: Double, momentum: Double): Unit ={
        // broadcast w
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        
        // compute full gradient
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ * this.nInv)
        
        //val gNorm: Double = g.map(a => a*a).sum
        //println("Driver: squared norm of gradient is " + gNorm.toString)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv
        
        // apply momentum
        for (j <- 0 until this.d) this.p(j) = this.p(j) * momentum + this.g(j)
        
        // take approximate Newton step
        for (j <- 0 until this.d) w(j) -= learningRate * this.p(j)
    }
    
}



