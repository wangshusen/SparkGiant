package distopt.quadratic.Common

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

/**
 * Constants common to driver and executors
 */
object Constants {
    val numStepSizes: Int = 10
    val baseStepSizes: Double = 4.0
}


/**
 * Solve a ridge regression problem. 
 * Model: 0.5*||X w - y||_2^2 + 0.5*gamma*||w||_2^2
 * 
 * @param sc SparkContext
 * @param n0 number of training samples
 * @param d0 number of features
 * @param m0 number of splits
 */
class Driver(sc: SparkContext, n0: Long, d0: Int, m0: Long) {
    // constants
    val n: Long = n0
    val d: Int = d0
    val m: Long = m0
    
    // variables
    var w: Array[Double] = new Array[Double](d)
    var g: Array[Double] = new Array[Double](d)
    var p: Array[Double] = new Array[Double](d)
    var trainError: Double = 0.0
    var objVal: Double = 0.0
    
    // for line search
    val numStepSizes: Int = Constants.numStepSizes
    val baseStepSizes: Double = Constants.baseStepSizes
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(baseStepSizes, _))
    
    /** 
     * Search for the best step size eta
     *
     * @param objVals array of objective values
     * @param pg = -0.1 * <p, g>
     * @return eta the biggest step size that leads to sufficient improvement
     */
    def lineSearch(objVals: Array[Double], pg: Double): Double = {
        var eta: Double = 0.0
        
        // backtracking line search (Armijo rule)
        for (j <- 0 until this.numStepSizes) {
            eta = this.stepSizes(j)
            var objValNew = objVals(j)
            // sufficient decrease in the objective value
            if (objValNew < this.objVal + pg * eta) { 
                return eta
            }
        }
        
        // if the search direction p does not lead to sufficient decrease,
        // then return the smallest step size in the candidate set.
        eta
    }
    
    /** 
     * Compute the mean squares error for test data.
     *
     * @param dataTest RDD of label-feature pair
     * @return mse of the test data
     */
    def predict(dataTest: RDD[(Double, Array[Double])]): Double = {
        val nTest: Long = dataTest.count
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        val error: Double = dataTest.map(pair => (pair._1, (pair._2, wBc.value).zipped.map(_ * _).sum))
                        .map(pair => (pair._1 - pair._2) * (pair._1 - pair._2))
                        .sum
        val mse: Double = error / nTest.toDouble
        mse
    }

}


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(var arr: Array[(Double, Array[Double])]) {
    // get data
    val s: Int = arr.size
    val d: Int = arr(0)._2.size
    val y: DenseMatrix[Double] = new DenseMatrix(s, 1, arr.map(pair => pair._1))
    val x: DenseMatrix[Double] = new DenseMatrix(d, s, arr.map(pair => pair._2).flatten)
    val xy: DenseMatrix[Double] = x * y
    val sDouble = this.s.toDouble

    // specific to training
    var gamma: Double = 0.0
    
    // for line search
    val numStepSizes: Int = Constants.numStepSizes
    val baseStepSizes: Double = Constants.baseStepSizes
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(baseStepSizes, _))
    val objValArray = new Array[Double](numStepSizes)

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
    }

    /**
     * Compute the local objective function values
     *      0.5*||X (w - eta*p) - y||_2^2 + 0.5*s*gamma*||(w - eta*p)||_2^2
     * for all eta in the candidate set.
     * This function is for line search.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def objFunVal(wArray: Array[Double], pArray: Array[Double]): Array[Double] = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        val p: DenseMatrix[Double] = new DenseMatrix(this.d, 1, pArray)
        var wTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var res: DenseMatrix[Double] = DenseMatrix.zeros[Double](s, 1)
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            res := this.x.t * wTmp - this.y
            var trainError: Double = res.toArray.map(a => a*a).sum
            var wNorm: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = (trainError + this.s * this.gamma * wNorm) / 2.0
        }
        
        this.objValArray
    }

    /**
     * Compute the local gradient.
     * As by-products, also compute the training error and objective value.
     *
     * @param w the current solution
     * @return g = X' * (X * w - y) + s * gamma * w , the local gradient
     * @return trainError = ||X w - y||_2^2 , the local training error
     * @return objVal = 0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2 , the local objective function value
     */
    def grad(wArray: Array[Double]): (Array[Double], Double, Double) = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        // gradient
        var res: DenseMatrix[Double] = this.x.t * w
        res := res - this.y
        var g: DenseMatrix[Double] = this.x * res 
        g := g + (this.s * this.gamma) * w
        // training error
        val trainError: Double = res.toArray.map(a => a*a).sum
        // objective function value
        val wNorm: Double = wArray.map(a => a*a).sum
        val objVal: Double = (trainError + this.s * this.gamma * wNorm) / 2
        (g.toArray, trainError, objVal)
    }

}
