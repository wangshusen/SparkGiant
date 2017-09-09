package distopt.logisticl1l2.Fista

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

import distopt.logisticl1l2._

/**
 * Solve a logistic regression problem using FISTA. 
 * Objective function is the mean of 
 * f_j (w) = log (1 + exp(-z_j)) + gamma1*||w||_1 + 0.5*gamma2*||w||_2^2, 
 * where z_j = <x_j, w>.
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])]) extends distopt.logisticl1l2.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = true
    val nInv: Double = 1.0 / n.toDouble
    
    var g: Array[Double] = new Array[Double](this.d)
    var p: Array[Double] = new Array[Double](this.d)
    var momentum: Double = 1.0
            
    // initialize executors
    val rdd: RDD[Common.Executor] = data.glom.map(new Common.Executor(_)).persist()
    println("There are " + rdd.count.toString + " partition.")

    /**
     * Train a logistic regression model.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param lipchitz inversely proportional to learning rate
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma1: Double, gamma2: Double, maxIter: Int, lipchitz: Double): (Array[Double], Array[Double], Array[Double]) = {
        this.gamma1 = gamma1
        this.gamma2 = gamma2
        
        // setup the executors for training
        val rddTrain: RDD[Common.Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma1, gamma2);
                                                 exe})
                                    .persist()
        rddTrain.count
        println("FISTA: gamma1 = " + gamma1.toString + ", gamma2 = " + gamma2.toString + ", lipchitz = " + lipchitz.toString)
        val t0: Double = System.nanoTime()
        
        // initialization
        this.momentum = 1.0
        for (j <- 0 until this.d) this.w(j) = 0.0
        for (j <- 0 until this.d) this.p(j) = this.w(j)
        
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIter)
        val objValArray: Array[Double] = new Array[Double](maxIter)
        val timeArray: Array[Double] = new Array[Double](maxIter)
        
        
        var t1: Double = System.nanoTime()
        
        for (t <- 0 until maxIter) {
            timeArray(t) = (t1 - t0) * 1.0E-9
            this.update2(rddTrain, lipchitz)
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
     * Update the gradient, objective value, and training error.
     */
    def updateGradient(rddTrain: RDD[Common.Executor]): Unit ={
        // broadcast p
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(this.p)
        
        // compute full gradient
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(pBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ * this.nInv)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv
    }

    /* Take one FISTA step.
     * Decompose the objective as f(w) = l(w) + r(w),
     *      where l(w) = logis(w),
     *      and   r(w) = gamma1*||w||_1 + 0.5*gamma2*||w||^2.
     *
     * Update:
     *  1. this.w
     *  2. this.p
     *  3. this.trainError
     *  4. this.objVal
     *
     * @param rddTrain RDD of executors
     * @param lipchitz inversely proportional to the learning rate
     * @return
     */
    def update1(rddTrain: RDD[Common.Executor], lipchitz: Double): Unit ={
        val momentumOld = this.momentum
        val wOld: Array[Double] = this.w.clone
        
        // gradient descent
        this.updateGradient(rddTrain)
        val z: Array[Double] = new Array[Double](this.d)
        for (j <- 0 until this.d) z(j) = this.p(j) - this.g(j) / lipchitz
        
        // the proximal operator: prox_{lambda * ||.||_1} (a)
        def l1shrinkage(a: Double, lambda: Double): Double = {
            if (a > lambda) return (a - lambda)
            else {
                if (a > -lambda) return 0
                else return (a + lambda)
            }
        }
        this.w = z.map(a => l1shrinkage(a, this.gamma1 / lipchitz)).map(a => a / (1 + this.gamma2 / lipchitz))
        this.momentum = (1 + math.sqrt(1 + 4 * momentumOld * momentumOld)) / 2
        val scaling: Double = (momentumOld - 1) / this.momentum
        for (j <- 0 until this.d) this.p(j) = this.w(j) + scaling * (this.w(j) - wOld(j))
    }

    /* Take one FISTA step.
     * Decompose the objective as f(w) = l(w) + r(w),
     *      where l(w) = logis(w) + 0.5*gamma2*||w||^2,
     *      and   r(w) = gamma1*||w||_1.
     *
     * Update:
     *  1. this.w
     *  2. this.p
     *  3. this.trainError
     *  4. this.objVal
     *
     * @param rddTrain RDD of executors
     * @param lipchitz inversely proportional to the learning rate
     * @return
     */
    def update2(rddTrain: RDD[Common.Executor], lipchitz: Double): Unit ={
        val momentumOld = this.momentum
        val wOld: Array[Double] = this.w.clone
        
        // gradient descent
        this.updateGradient(rddTrain)
        val z: Array[Double] = new Array[Double](this.d)
        for (j <- 0 until this.d) z(j) = this.p(j) - this.g(j) / lipchitz
        
        // the proximal operator: prox_{lambda * ||.||_1} (a)
        def l1shrinkage(a: Double, lambda: Double): Double = {
            if (a > lambda) return (a - lambda)
            else {
                if (a > -lambda) return 0
                else return (a + lambda)
            }
        }
        this.w = z.map(a => l1shrinkage(a, this.gamma1 / lipchitz))
        this.momentum = (1 + math.sqrt(1 + 4 * momentumOld * momentumOld)) / 2
        val scaling: Double = (momentumOld - 1) / this.momentum
        for (j <- 0 until this.d) this.p(j) = this.w(j) + scaling * (this.w(j) - wOld(j))
    }
}


