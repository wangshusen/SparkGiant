package distopt.logistic.Common

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
 * Solve a logistic regression problem. 
 * Model: mean( log( 1 + exp(-z_j) ) ) + 0.5*gamma*||w||_2^2,
 *        where z_j = <x_j, w>.
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
     * Compute the error for test data.
     *
     * @param dataTest RDD of label-feature pair
     * @return average of the 0-1 error of test data
     */
    def predict(dataTest: RDD[(Double, Array[Double])]): Double = {
        val nTest: Long = dataTest.count
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        val error: Double = dataTest.map(pair => (pair._1, (pair._2, wBc.value).zipped.map(_ * _).sum))
                                .map(pair => -1.0 * pair._1 * pair._2)
                                .filter(_ > -0.1)
                                .count
        error / nTest.toDouble
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
    val x: DenseMatrix[Double] = new DenseMatrix(d, s, arr.map(pair => pair._2.map(a => pair._1 * a)).flatten)
    val y: Array[Double] = arr.map(pair => pair._1)
    val a: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, s)
    val sDouble = this.s.toDouble
    
    // make sure y has {+1, -1} values
    val ymin: Double = y.min
    assert(ymin > -1.01 && ymin < -0.99)
    val ymax: Double = y.max
    assert(ymax > 0.99 && ymax < 1.01)

    // specific to training
    var gamma: Double = 0.0
    
    // parameters for CG
    var q: Int = 0 // number of CG iterations
    var isFormHessian: Boolean = false
    var cgtol: Double = 1e-16 / this.sDouble
            
    def setParam(q0: Int, isFormHessian0: Boolean){
        this.q = q0
        this.isFormHessian = isFormHessian0
        cgtol = 1e-16 / this.sDouble
    }
    
    // for line search
    val numStepSizes: Int = Constants.numStepSizes
    val baseStepSizes: Double = Constants.baseStepSizes
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(baseStepSizes, _))
    val objValArray = new Array[Double](numStepSizes)

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
        cgtol = 1e-16 / this.sDouble
    }

    /**
     * Compute the sum local gradient of the objective function
     *      f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
     *      where z_j = <x_j, w-eta*p>.
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
        
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            val zexp: Array[Double] = (this.x.t * wTmp).toArray.map((a: Double) => math.exp(a))
            val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
            val wNorm: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = loss + this.s * this.gamma * wNorm * 0.5
        }
        
        this.objValArray
    }

    /**
     * Compute the sum local gradient of the objective function
     *      f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
     *      where z_j = <x_j, w>.
     * As by-products, also compute the training error and objective value.
     *
     * @param w the current solution
     * @return g sum of the gradients of f_j (w) for all the local data.
     * @return trainError sum of (y_j != sign(z_j))
     * @return objVal sum of f_j (2) for all the local data.
     */
    def grad(wArray: Array[Double]): (Array[Double], Double, Double) = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        val z: Array[Double] = (this.x.t * w).toArray
        val zexp: Array[Double] = z.map((a: Double) => math.exp(a))
        // gradient
        val c: DenseMatrix[Double] = new DenseMatrix(this.s, 1, zexp.map((a: Double) => -1.0 / (1.0 + a)))
        val g: DenseMatrix[Double] = this.x * c + (this.s * this.gamma) * w
        // objective function value
        val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
        val wNorm: Double = wArray.map(a => a*a).sum
        val objVal: Double = loss + this.s * this.gamma * wNorm * 0.5
        // training error
        val pred: Array[Double] = z.map((a: Double) => math.signum(a))
        val trainError: Double = (pred, this.y).zipped.map(-1.0 * _ * _).filter(_ > -0.1).length
        
        (g.toArray, trainError, objVal)
    }

    
    /**
     * Solve logistic regression by SVRG.
     * Objective function is the mean of 
     * f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
     * where z_j = <x_j, w>.
     *
     * @param learningrate learning rate (step size) of gradient descent
     * @param q maximum number of iterations
     * @return w the trained model
     */
    def solve(learningrate: Double, q: Int): Array[Double] = {
        distopt.utils.Logistic.svrgSolver(this.x, this.gamma, learningrate, q).map(_ * this.sDouble)
    }
}
