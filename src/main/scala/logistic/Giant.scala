package distopt.logistic.Giant

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a logistic regression problem using GIANT with the local problems inexactly solved by CG. 
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
    val isMute: Boolean = false
            
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()
    //println("There are " + rdd.count.toString + " partition.")
    //println("Driver: executors are initialized using the input data!")

    /**
     * Train a logistic regression model using GIANT with the local problems solved by fixed number of CG steps.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param q number of CG iterations
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, q: Int): (Array[Double], Array[Double], Array[Double]) = {
        // decide whether to form the Hessian matrix
        val s: Double = this.n.toDouble / this.m.toDouble
        val cost1: Double = 4 * q * s // CG without the Hessian formed
        val cost2: Double = (s + q) * this.d // CG with the Hessian formed
        var isFormHessian: Boolean = if(cost1 < cost2) false else true
        
        val t0: Double = System.nanoTime()
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);  
                                                 exe.setLocalMaxIter(q); 
                                                 exe.setIsFormHessian(isFormHessian);
                                                 exe})
                                    .persist()
        println("Driver: executors are setup for training! gamma = " + gamma.toString + ", q = " + q.toString + ", isFormHessian = " + isFormHessian.toString)
        
        
        // initialize w by model averaging
        if (isModelAvg) {
            val svrgLearningRate: Double = 1.0
            this.w = rddTrain.map(_.solve(svrgLearningRate, q))
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
        // compute full gradient
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        var tmp: (Array[Double], Double, Double) = rddTrain.map(exe => exe.grad(wBc.value))
                    .reduce((a, b) => ((a._1,b._1).zipped.map(_ + _), a._2+b._2, a._3+b._3))
        this.g = tmp._1.map(_ * this.nInv)
        val gBc: Broadcast[Array[Double]] = this.sc.broadcast(this.g)
        
        this.gNorm = g.map(a => a*a).sum
        //println("Driver: squared norm of gradient is " + this.gNorm.toString)
        
        // update the training error and objective value
        this.trainError = tmp._2 * this.nInv
        this.objVal = tmp._3 * this.nInv

        // compute the averaged Newton direction
        this.p = rddTrain.map(exe => exe.newton(wBc.value, gBc.value))
                        .reduce((a,b) => (a,b).zipped.map(_ + _)) 
                        .map(_ * this.nInv)
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
            eta = this.lineSearch(objVals, -0.1 * pg)
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
    val cg: distopt.utils.CG = new distopt.utils.CG(this.d)
    
    var isFormHessian: Boolean = false
    def setIsFormHessian(isFormHessian0: Boolean){
        this.isFormHessian = isFormHessian0
    }
    
    /**
     * Compute approximate Newton direction using the local data.
     *
     * @param wArray the current solution
     * @param gArray the full gradient
     * @return p the Newton direction
     */
    def newton(wArray: Array[Double], gArray: Array[Double]): Array[Double] = {
        val g: DenseVector[Double] = new DenseVector(gArray)
        val w: DenseVector[Double] = new DenseVector(wArray)
        var p: Array[Double] = Array.empty[Double]
        val zexp: Array[Double] = (this.x.t * w).toArray.map((a: Double) => math.exp(a))
        val ddiag: Array[Double] = zexp.map((r: Double) => (math.sqrt(r) / (1.0 + r)))
        for (j <- 0 until this.s) {
            this.a(::, j) := ddiag(j) * this.x(::, j)
        }
        
        if (this.isFormHessian) {
            val aa: DenseMatrix[Double] = this.a * this.a.t
            p = cg.solver2(aa, this.sDouble * g, this.sDouble * this.gamma, this.q)
        }
        else {
            p = cg.solver1(this.a, this.sDouble * g, this.sDouble * this.gamma, this.q)
        }
        p.map((a: Double) => a * this.sDouble)
    }
        
}


