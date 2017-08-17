package distopt.quadratic.Admm

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._

/**
 * Solve a ridge regression problem using GIANT with the local problems solved by CG. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param data RDD of (label, feature)
 * @param isModelAvg is true if model averaging is used to initialize w
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isModelAvg: Boolean = false)
        extends distopt.quadratic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = true
    var rho: Double = 1.0
    var gamma: Double = 1.0
    val mInv: Double = 1.0 / this.m
    
    // initialize executors
    val rdd: RDD[Executor] = data.glom.zipWithIndex.map(pair => new Executor(pair._1, pair._2)).persist()
            
    // assert that the indices of executors are 0, ..., m-1 without gap
    val indexArray: Array[Int] = rdd.map(exe => exe.index).collect.sortWith(_ < _)
    for (i <- 0 until this.m.toInt) assert(i == indexArray(i))
    
    var u: Array[Double] = new Array[Double](d)
    val aArrays: Array[Array[Double]] = Array.ofDim[Double](this.m.toInt, this.d)
    var wArrays: Array[Array[Double]] = Array.ofDim[Double](this.m.toInt, this.d)

    /**
     * Train a ridge regression model using GIANT with the local problems solved by fixed number of CG steps.
     *
     * @param gamma the regularization parameter
     * @param maxIter max number of iterations
     * @param q number of CG iterations
     * @return trainErrorArray the training error in each iteration
     * @return objValArray the objective values in each iteration
     * @return timeArray the elapsed times counted at each iteration
     */
    def train(gamma: Double, maxIter: Int, q: Int): (Array[Double], Array[Double], Array[Double]) = {
        this.gamma = gamma
        this.rho = 0.1 * gamma
        
        // decide whether to form the Hessian matrix
        val s: Double = this.n.toDouble / this.m.toDouble
        val cost1: Double = 0.2 * maxIter * q * s // CG without the Hessian formed
        val cost2: Double = s * this.d + 0.2 * maxIter * q * this.d // CG with the Hessian formed
        var isFormHessian: Boolean = if(cost1 < cost2) false else true
        
        // setup the executors for training
        val rddTrain: RDD[Executor] = this.rdd
                                    .map(exe => {exe.setGamma(gamma);  
                                                 exe.setMaxInnerIter(q); 
                                                 exe.setFormHessian(isFormHessian);
                                                 exe})
                                    .persist()
        rddTrain.count
        val t0: Double = System.nanoTime()
        
        // initialize a, w, u 
        aArrays.map(ai => ai.map(aij => 0.0))
        wArrays.map(wi => wi.map(wij => 0.0))
        if (this.isModelAvg) {
            this.u = rddTrain.map(_.solve())
                            .reduce((a,b) => (a,b).zipped.map(_ + _))
                            .map(_ / this.n.toDouble)
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
        
        //for (j <- 0 until this.d) this.w(j) = this.u(j)
        
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
        val wAvgBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        
        // update the objective value and training error
        val (ov, te): (Double, Double) = rddTrain
                        .map(exe => exe.objs(uBc.value))
                        //.map(exe => exe.objs(wAvgBc.value))
                        .reduce((pair1, pair2) => (pair1._1+pair2._1, pair1._2+pair2._2))
        this.objVal = ov * this.nInv
        this.trainError = te * this.nInv
        
        // update w
        val tmp: Array[(Int, Array[Double])] = rddTrain
                        .map(exe => exe.updateW(aBc.value, uBc.value, wBc.value, rhoBc.value))
                        .collect
        this.wArrays = tmp.sortWith(_._1 < _._1)
                        .map((pair: (Int, Array[Double])) => pair._2)
        
        // update u (locally)
        val normalizer: Double = this.rho / (this.gamma + this.rho)
        this.w = this.wArrays.reduce(this.arrayAdd).map(_ * mInv)
        val aAvg: Array[Double] = this.aArrays.reduce(this.arrayAdd).map(_ * mInv)
        for (j <- 0 until this.d) this.u(j) = (this.w(j) + aAvg(j)) * normalizer
        
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
        distopt.quadratic.Common.Executor(arr) {
    // constants
    val index: Int = idx.toInt
      
    /**
     * Locally update w_i.
     * As by-products, compute the objective value and training error.
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
        val u: DenseVector[Double] = new DenseVector(uArray)
        val srho = this.sDouble * rho
        val diff: DenseVector[Double] = srho * (u - a)
        
        // update w
        var wnew: Array[Double] = Array.empty[Double]
        if (this.isFormHessian) {
            wnew = cg.solver2(this.xx, this.xy + diff, srho, this.q, wold) 
        }
        else {
            wnew= cg.solver1(this.x, this.xy + diff, srho, this.q, wold)
        }
        
        
        (this.index, wnew)
    }
    
    /**
     * Compute local objective value and training error.
     */
    def objs(wArray: Array[Double]): (Double, Double) = {
        val w: DenseVector[Double] = new DenseVector(wArray)
        var res: DenseVector[Double] = this.x.t * w - this.y
        val trainError: Double = res.toArray.map(a => a*a).sum
        val wNorm: Double = wArray.map(a => a*a).sum
        val objVal: Double = (trainError + this.sDouble * this.gamma * wNorm) / 2
        (objVal, trainError)
    }   
}


