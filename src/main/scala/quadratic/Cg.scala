package distopt.quadratic.Cg

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a ridge regression problem using conjugate gradient (CG) method. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])])
        extends distopt.quadratic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    println("Driver: executors are initialized using the input data!")

    /**
     * Train a ridge regression model using GIANT with the local problems solved by fixed number of CG steps.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, isModelAvg: Boolean = false): (Array[Double], Array[Double], Array[Double]) = {
        // decide whether to form the Hessian matrix
        val s: Double = this.n.toDouble / this.m.toDouble
        val cost1: Double = 0.2 * maxIter * q * s // CG without the Hessian formed
        val cost2: Double = s * this.d + 0.2 * maxIter * q * this.d // CG with the Hessian formed
        var isFormHessian: Boolean = if(cost1 < cost2) false else true
        
        println("There are " + this.rdd.count.toString + " executors.")
        val t0: Double = System.nanoTime()
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);
                                                 exe.setFormHessian(isFormHessian);
                                                 exe})
                                    .persist()
        println("count = " + rddTrain.count.toString)
        println("Driver: executors are setup for training! gamma = " + gamma.toString + ", q = " + q.toString + ", isFormHessian = " + isFormHessian.toString)
        
        // initialize w by model averaging
        if (isModelAvg) {
            this.w = rddTrain.map(_.solve())
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ / this.n.toDouble)
            println("Driver: model averaging is done!")
        }
        else {
            for (j <- 0 until this.d) this.w(j) = 0
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
            if (this.gNorm < this.gNormTol) {
                return (trainErrorArray.slice(0, t+1), 
                        objValArray.slice(0, t+1), 
                        timeArray.slice(0, t+1))
            }
        }
        
        (trainErrorArray, objValArray, timeArray)
    }

    /**
     * Take one approximate Newton step.
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
        // compute gradient, objective value, and training errors
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        
        
        
        this.updateGrad(wBc, rddTrain)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        

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
            var pg: Double = 0.0
            for (j <- 0 until this.d) pg += this.p(j) * this.g(j)
            eta = this.lineSearch(objVals, -0.1 * pg)
            //println("Eta = " + eta.toString)
        }
        
        // update w by taking approximate Newton step
        for (j <- 0 until this.d) this.w(j) -= eta * this.p(j)
    }
            
    /**
     * Update this.g, this.gNorm, this.trainError, this.objVal
     *
     * @param wBc broadcast of this.w
     * @param rddTrain RDD of training samples
     * @return
     */
    def updateGrad(wBc: Broadcast[Array[Double]], rddTrain: RDD[Executor]): Unit = {
        // compute full gradient
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ / this.n.toDouble)
        this.gNorm = g.map(a => a*a).sum
        //println("Squared norm of gradient is " + gNorm.toString)
        
        // update the training error and objective value
        this.trainError = tmp._2 * (1.0 / this.n)
        this.objVal = tmp._3 * (1.0 / this.n)
    }
    
}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(arr: Array[(Double, Array[Double])]) extends 
        distopt.quadratic.Common.Executor(arr) {
    /**
     * Compute X * X' * p.
     * Here X is d-by-s and p is d-by-1.
     */
    def xxByP(pArray: Array[Double]): Array[Double] = {
        val p: DenseVector[Double] = new DenseVector(pArray)
        val xp: DenseVector[Double] = this.x.t * p
        (this.x * xp).toArray
    }
    /**
     * Compute the local Newton direction
     *
     * @param gArray the full gradient
     * @return the local Newton direction scaled by s
     */
    def newton(gArray: Array[Double]): Array[Double] = {
        val g: DenseVector[Double] = new DenseVector(gArray)
        var p: Array[Double] = Array.empty[Double]
        if (this.isFormHessian) {
            p = cg.solver2(this.xx, this.sDouble * g, this.sDouble * this.gamma, this.q)
        }
        else {
            p = cg.solver1(this.x, this.sDouble * g, this.sDouble * this.gamma, this.q)
        }
        p.map((a: Double) => a * this.sDouble)
    }
}


