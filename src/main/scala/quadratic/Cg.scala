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
 * @param isModelAvg is true if model averaging is used to initialize w
 */
class Driver(sc: SparkContext, data: RDD[(Double, Array[Double])], isModelAvg: Boolean = false)
        extends distopt.quadratic.Common.Driver(sc, data.count, data.take(1)(0)._2.size, data.getNumPartitions) {
    val isMute: Boolean = true
            
    // initialize executors
    val rdd: RDD[Executor] = data.glom.map(new Executor(_)).persist()

    /**
     * Train a ridge regression model using GIANT with the local problems solved by fixed number of CG steps.
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
                                                 exe})
                                    .persist()
        rddTrain.count
        val t0: Double = System.nanoTime()
        
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
        
        // setup buffers
        val ngamma = this.n * gamma
        val xy: Array[Double] = rddTrain.map(_.xy.toArray)
                                    .reduce((a,b) => (a,b).zipped.map(_ + _))
        val r: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val p: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val ap: DenseVector[Double] = DenseVector.zeros[Double](this.d)
        val wnew: DenseVector[Double] = new DenseVector(this.w)
        val xxw: DenseVector[Double] = new DenseVector(this.xxByP(this.w, rddTrain))
        r := (new DenseVector(xy)) - ngamma * wnew - xxw
        p := r
        var rsold: Double = r.toArray.map(x => x*x).sum
        var rsnew: Double = 0.0
        var alpha: Double = 0.0
        
        // record the objectives of each iteration
        val trainErrorArray: Array[Double] = new Array[Double](maxIter)
        val objValArray: Array[Double] = new Array[Double](maxIter)
        val timeArray: Array[Double] = new Array[Double](maxIter)
        
        var t1: Double = System.nanoTime()
        var i: Int = -1
        
        for (t <- 0 until maxIter) {
            var xxp: DenseVector[Double] = new DenseVector(this.xxByP(p.toArray, rddTrain))
            ap := ngamma * p + xxp
            var pap: Double = 0.0
            for (j <- 0 until d) pap += p(j) * ap(j)
            alpha = rsold / pap
            wnew += alpha * p
            r -= alpha * ap
            rsnew = r.toArray.map(a => a*a).sum
            
            if (t % 2 == 0) { // record the objective values and training errors
                i += 1
                timeArray(i) = (t1 - t0) * 1.0E-9
                t1 = System.nanoTime()
                this.objs(wnew.toArray, rddTrain)
                trainErrorArray(i) = this.trainError
                objValArray(i) = this.objVal
                if (!this.isMute) println("Iteration " + t.toString + ":\t objective value is " + this.objVal.toString + ",\t time: " + timeArray(t).toString)
            }
            if (rsnew < this.gNormTol) {
                return (trainErrorArray.slice(0, i+1), 
                        objValArray.slice(0, i+1), 
                        timeArray.slice(0, i+1))
            }
            p *= rsnew / rsold
            p += r
            rsold = rsnew
        }
        
        (trainErrorArray, objValArray, timeArray)
    }
    
    /**
     * Compute X * X' * p, where X is d-by-n and p is d-by-1.
     */
    def xxByP(pArray: Array[Double], rddTrain: RDD[Executor]): Array[Double] = {
        val pBc: Broadcast[Array[Double]] = this.sc.broadcast(pArray)
        val xxp: Array[Double] = rddTrain.map(_.xxByP(pBc.value))
                                    .reduce((a,b) => (a,b).zipped.map(_ + _))
        xxp
    }

    /**
     * Compute the objective value and training error.
     */
    def objs(wArray: Array[Double], rddTrain: RDD[Executor]): Unit = {
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(wArray)
        val tmp: (Double, Double) = rddTrain.map(_.objs(wBc.value))
                                    .reduce((a,b) => (a._1+b._1, a._2+b._2))
        this.objVal = tmp._1 * this.nInv
        this.trainError = tmp._2 * this.nInv
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
     * Compute the local objective value and training error.
     */
    def objs(wArray: Array[Double]): (Double, Double) = {
        val w: DenseVector[Double] = new DenseVector(wArray)
        // training error
        var res: DenseVector[Double] = this.x.t * w - this.y
        val trainError: Double = res.toArray.map(a => a*a).sum
        // objective function value
        val wNorm: Double = wArray.map(a => a*a).sum
        val objVal: Double = (trainError + this.sDouble * this.gamma * wNorm) / 2
        (objVal, trainError)
    }
            
}


